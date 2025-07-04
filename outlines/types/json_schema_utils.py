"""Utilities for handling JSON schema compatibility."""

import json
import hashlib
import threading
import time
import zlib
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Union, Set, Optional


# Configuration class for better maintainability
class JsonSchemaConfig:
    """Configuration for JSON schema preprocessing."""

    def __init__(self):
        self.max_cache_size: int = 1000
        self.max_recursion_depth: int = 50
        self.max_preprocessing_depth: int = 100
        self.max_schema_size_bytes: int = 10 * 1024 * 1024  # 10MB
        self.enable_fast_hashing: bool = True
        self.enable_compression: bool = True
        self.enable_fallback: bool = True
        self.enable_metrics: bool = True


# Global configuration instance
_CONFIG = JsonSchemaConfig()


# Performance metrics
class PreprocessingMetrics:
    """Thread-safe metrics collection."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.preprocessing_count = 0
        self.preprocessing_errors = 0
        self.total_processing_time = 0.0
        self.fallback_count = 0

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self):
        with self._lock:
            self.cache_misses += 1

    def record_preprocessing(self, processing_time: float, success: bool = True):
        with self._lock:
            self.preprocessing_count += 1
            self.total_processing_time += processing_time
            if not success:
                self.preprocessing_errors += 1

    def record_fallback(self):
        with self._lock:
            self.fallback_count += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                self.cache_hits / total_requests if total_requests > 0 else 0.0
            )
            avg_processing_time = (
                self.total_processing_time / self.preprocessing_count
                if self.preprocessing_count > 0
                else 0.0
            )

            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "preprocessing_count": self.preprocessing_count,
                "preprocessing_errors": self.preprocessing_errors,
                "total_processing_time": self.total_processing_time,
                "avg_processing_time": avg_processing_time,
                "fallback_count": self.fallback_count,
            }


# Global metrics instance
_METRICS = PreprocessingMetrics()

# Thread-safe LRU cache for preprocessed schemas
# Store compressed JSON for memory efficiency
_SCHEMA_CACHE: OrderedDict[str, bytes] = OrderedDict()
_cache_lock = threading.Lock()


def preprocess_schema_for_union_types(schema: Union[str, dict]) -> str:
    """
    Preprocess a JSON schema to handle union types (array type specifications).

    This function converts type arrays like ["string", "null"] into anyOf format that
    outlines-core 0.1.26 can handle. It includes optimization to skip schemas that
    don't contain type arrays and caching for repeated schemas.

    Parameters
    ----------
    schema : Union[str, dict]
        The JSON schema as a string or dictionary.

    Returns
    -------
    str
        The preprocessed JSON schema as a string.

    Raises
    ------
    ValueError
        If the schema is invalid JSON or contains unsupported structures.
    """
    # Convert to string for hashing and processing
    if isinstance(schema, str):
        schema_str = schema
        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema: {e}")
    else:
        schema_dict = schema
        schema_str = json.dumps(
            schema, sort_keys=True, separators=(",", ":")
        )  # Compact for performance

    # Basic validation to catch obviously invalid schemas early
    if not validate_json_schema_basic(schema_dict):
        raise ValueError("Schema does not appear to be a valid JSON schema")

    # Input validation for DoS protection
    if len(schema_str.encode("utf-8")) > _CONFIG.max_schema_size_bytes:
        raise ValueError(
            f"Schema size exceeds maximum allowed size of {_CONFIG.max_schema_size_bytes} bytes"
        )

    # Performance optimization: use faster hashing for cache keys
    if _CONFIG.enable_fast_hashing:
        # CRC32 is much faster than MD5 for our use case
        schema_hash = str(zlib.crc32(schema_str.encode("utf-8")))
    else:
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()

    # Check cache first (thread-safe)
    with _cache_lock:
        if schema_hash in _SCHEMA_CACHE:
            # Move to end (most recently used) for LRU
            compressed_value = _SCHEMA_CACHE.pop(schema_hash)
            _SCHEMA_CACHE[schema_hash] = compressed_value

            # Decompress and return cached result
            if _CONFIG.enable_compression:
                cached_result = zlib.decompress(compressed_value).decode("utf-8")
            else:
                cached_result = compressed_value.decode("utf-8")

            if _CONFIG.enable_metrics:
                _METRICS.record_cache_hit()
            return cached_result

    # Cache miss
    if _CONFIG.enable_metrics:
        _METRICS.record_cache_miss()

    # Quick check: if schema doesn't contain any type arrays, return as-is
    if not _contains_type_arrays(schema_dict, max_depth=_CONFIG.max_recursion_depth):
        _update_cache_threadsafe(schema_hash, schema_str)
        return schema_str

    # Process the schema with timing
    start_time = time.time()
    try:
        preprocessed = _preprocess_schema_dict_safe(
            schema_dict, max_depth=_CONFIG.max_preprocessing_depth
        )
        result = json.dumps(
            preprocessed, separators=(",", ":")
        )  # Compact JSON for performance
        processing_time = time.time() - start_time

        _update_cache_threadsafe(schema_hash, result)

        if _CONFIG.enable_metrics:
            _METRICS.record_preprocessing(processing_time, success=True)

        return result
    except Exception as e:
        processing_time = time.time() - start_time

        if _CONFIG.enable_metrics:
            _METRICS.record_preprocessing(processing_time, success=False)

        # Graceful degradation: fallback to original schema if enabled
        if _CONFIG.enable_fallback:
            if _CONFIG.enable_metrics:
                _METRICS.record_fallback()
            return schema_str
        else:
            # Preserve error context and add schema information
            schema_preview = (
                str(schema_dict)[:200] + "..."
                if len(str(schema_dict)) > 200
                else str(schema_dict)
            )
            raise ValueError(
                f"Error preprocessing schema: {e}. Schema preview: {schema_preview}"
            ) from e


def _contains_type_arrays(obj: Any, max_depth: int = 50) -> bool:
    """
    Quick check to determine if a schema contains any type arrays.

    This optimization avoids expensive recursive processing for schemas
    that don't need preprocessing. Includes early termination, depth limiting,
    and circular reference detection.

    Parameters
    ----------
    obj : Any
        The object to check for type arrays
    max_depth : int, optional
        Maximum recursion depth to prevent stack overflow, by default 50

    Returns
    -------
    bool
        True if type arrays are found, False otherwise
    """
    visited = set()  # Track visited object IDs to detect circular references

    def _recursive_check(obj: Any, depth: int) -> bool:
        if depth > max_depth:
            # Avoid stack overflow on very deep structures
            return False

        # Circular reference detection for mutable objects
        if isinstance(obj, (dict, list)):
            obj_id = id(obj)
            if obj_id in visited:
                # Circular reference detected, skip to avoid infinite loop
                return False
            visited.add(obj_id)

        try:
            if isinstance(obj, dict):
                # Check if this object has a type field that's an array (early termination)
                if "type" in obj and isinstance(obj["type"], list):
                    return True
                # Recursively check all values, with early termination
                for v in obj.values():
                    if _recursive_check(v, depth + 1):
                        return True  # Early termination on first match
                return False
            elif isinstance(obj, list):
                # Recursively check all items, with early termination
                for item in obj:
                    if _recursive_check(item, depth + 1):
                        return True  # Early termination on first match
                return False
            else:
                return False
        finally:
            # Remove from visited set when backtracking
            if isinstance(obj, (dict, list)):
                visited.discard(id(obj))

    return _recursive_check(obj, 0)


def _update_cache_threadsafe(key: str, value: str) -> None:
    """Update the schema cache with LRU eviction (thread-safe)."""
    global _SCHEMA_CACHE
    with _cache_lock:
        # Remove oldest entry if cache is full (LRU eviction)
        if len(_SCHEMA_CACHE) >= _CONFIG.max_cache_size:
            _SCHEMA_CACHE.popitem(last=False)  # Remove oldest (first) item

        # Compress value for memory efficiency if enabled
        if _CONFIG.enable_compression:
            compressed_value = zlib.compress(value.encode("utf-8"))
        else:
            compressed_value = value.encode("utf-8")

        _SCHEMA_CACHE[key] = compressed_value


def _preprocess_schema_dict_safe(obj: Any, max_depth: int = 100) -> Any:
    """
    Safely preprocess a schema dictionary to convert type arrays.

    This function handles all JSON schema keywords and preserves the schema
    structure while converting type arrays to anyOf format. Includes circular
    reference detection and depth limiting.

    Parameters
    ----------
    obj : Any
        The object to preprocess.
    max_depth : int, optional
        Maximum recursion depth to prevent stack overflow, by default 100

    Returns
    -------
    Any
        The preprocessed object.

    Raises
    ------
    ValueError
        If circular references are detected or maximum depth is exceeded.
    """
    visited = set()  # Track visited object IDs to detect circular references

    def _recursive_preprocess(obj: Any, depth: int, path: str = "") -> Any:
        if depth > max_depth:
            raise ValueError(
                f"Maximum schema depth ({max_depth}) exceeded at path: {path}"
            )

        # Circular reference detection for mutable objects
        if isinstance(obj, (dict, list)):
            obj_id = id(obj)
            if obj_id in visited:
                raise ValueError(
                    f"Circular reference detected in schema at path: {path}"
                )
            visited.add(obj_id)

        try:
            if isinstance(obj, dict):
                # Check if this object has a type field that's an array
                if "type" in obj and isinstance(obj["type"], list):
                    result = _convert_type_array_to_anyof_safe(obj, depth, path)
                    return result
                else:
                    # Recursively process all values
                    return {
                        k: _recursive_preprocess(
                            v, depth + 1, f"{path}.{k}" if path else k
                        )
                        for k, v in obj.items()
                    }
            elif isinstance(obj, list):
                # Recursively process all items in lists
                return [
                    _recursive_preprocess(item, depth + 1, f"{path}[{i}]")
                    for i, item in enumerate(obj)
                ]
            else:
                # Return other types as-is
                return obj
        finally:
            # Remove from visited set when backtracking
            if isinstance(obj, (dict, list)):
                visited.discard(id(obj))

    return _recursive_preprocess(obj, 0)


def _preprocess_schema_dict(obj: Any) -> Any:
    """
    Legacy wrapper for backward compatibility.

    Recursively preprocess a schema dictionary to convert type arrays.

    This function handles all JSON schema keywords and preserves the schema
    structure while converting type arrays to anyOf format.

    Parameters
    ----------
    obj : Any
        The object to preprocess.

    Returns
    -------
    Any
        The preprocessed object.
    """
    return _preprocess_schema_dict_safe(obj)


def _convert_type_array_to_anyof_safe(
    obj: Dict[str, Any], depth: int, path: str
) -> Dict[str, Any]:
    """
    Safely convert a schema object with type array to anyOf format.

    This function handles all JSON schema keywords and ensures they are
    properly distributed to the appropriate type alternatives. Includes
    validation and error context.
    """
    try:
        return _convert_type_array_to_anyof(obj)
    except Exception as e:
        raise ValueError(
            f"Error converting type array to anyOf at path '{path}' (depth {depth}): {e}"
        ) from e


def _convert_type_array_to_anyof(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a schema object with type array to anyOf format.

    This function handles all JSON schema keywords and ensures they are
    properly distributed to the appropriate type alternatives.
    """
    type_array = obj["type"]
    any_of = []

    # Create a new object without the type field
    base_obj = {k: v for k, v in obj.items() if k != "type"}

    # Keywords that should be copied to all type alternatives
    global_keywords = {"title", "description", "default", "$id", "$schema", "$ref"}

    # Keywords that are type-specific and should only go with relevant types
    type_specific_keywords = {
        "string": {"minLength", "maxLength", "pattern", "format"},
        "number": {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        },
        "integer": {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        },
        "array": {"items", "minItems", "maxItems", "uniqueItems", "additionalItems"},
        "object": {
            "properties",
            "required",
            "additionalProperties",
            "minProperties",
            "maxProperties",
            "patternProperties",
            "dependencies",
        },
        "boolean": set(),
        "null": set(),
    }

    # Keywords that should be preserved at the anyOf level
    anyof_level_keywords = global_keywords.union({"enum", "const", "examples"})

    for type_str in type_array:
        if type_str == "null":
            any_of.append({"type": "null"})
        else:
            # Start with the base type
            type_obj = {"type": type_str}

            # Add type-specific constraints
            relevant_keywords = type_specific_keywords.get(type_str, set())
            for key in relevant_keywords:
                if key in base_obj:
                    # Recursively process nested structures
                    if key in {
                        "properties",
                        "items",
                        "additionalItems",
                        "patternProperties",
                        "dependencies",
                    }:
                        type_obj[key] = _preprocess_schema_dict(base_obj[key])
                    else:
                        type_obj[key] = base_obj[key]

            any_of.append(type_obj)

    # Create the result with anyOf
    result = {"anyOf": any_of}

    # Add global keywords at the anyOf level
    for key in anyof_level_keywords:
        if key in base_obj:
            if key in {"enum", "const", "examples"}:
                # These should be recursively processed
                result[key] = _preprocess_schema_dict(base_obj[key])
            else:
                result[key] = base_obj[key]

    # Add any remaining keywords that we haven't categorized
    # This ensures we don't lose any schema information
    categorized_keywords = anyof_level_keywords.union(
        set().union(*type_specific_keywords.values())
    )
    for key in base_obj:
        if key not in categorized_keywords:
            result[key] = _preprocess_schema_dict(base_obj[key])

    return result


def clear_schema_cache() -> None:
    """Clear the schema preprocessing cache (thread-safe)."""
    global _SCHEMA_CACHE
    with _cache_lock:
        _SCHEMA_CACHE.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive statistics about the schema cache and preprocessing (thread-safe)."""
    with _cache_lock:
        cache_stats = {
            "cache_size": len(_SCHEMA_CACHE),
            "max_cache_size": _CONFIG.max_cache_size,
            "oldest_key": next(iter(_SCHEMA_CACHE)) if _SCHEMA_CACHE else None,
            "newest_key": next(reversed(_SCHEMA_CACHE)) if _SCHEMA_CACHE else None,
        }

        # Calculate cache memory usage (approximate)
        if _SCHEMA_CACHE:
            total_compressed_bytes = sum(len(value) for value in _SCHEMA_CACHE.values())
            cache_stats["total_cache_memory_bytes"] = total_compressed_bytes
            cache_stats["avg_cache_entry_bytes"] = total_compressed_bytes / len(
                _SCHEMA_CACHE
            )
        else:
            cache_stats["total_cache_memory_bytes"] = 0
            cache_stats["avg_cache_entry_bytes"] = 0

    # Add preprocessing metrics if enabled
    if _CONFIG.enable_metrics:
        metrics_stats = _METRICS.get_stats()
        cache_stats.update(metrics_stats)

    return cache_stats


def configure_preprocessing(
    max_cache_size: Optional[int] = None,
    max_recursion_depth: Optional[int] = None,
    max_preprocessing_depth: Optional[int] = None,
    max_schema_size_bytes: Optional[int] = None,
    enable_fast_hashing: Optional[bool] = None,
    enable_compression: Optional[bool] = None,
    enable_fallback: Optional[bool] = None,
    enable_metrics: Optional[bool] = None,
) -> None:
    """
    Configure JSON schema preprocessing behavior.

    Parameters
    ----------
    max_cache_size : int, optional
        Maximum number of schemas to cache
    max_recursion_depth : int, optional
        Maximum recursion depth for type array detection
    max_preprocessing_depth : int, optional
        Maximum recursion depth for schema preprocessing
    max_schema_size_bytes : int, optional
        Maximum schema size in bytes (DoS protection)
    enable_fast_hashing : bool, optional
        Use fast CRC32 hashing instead of MD5
    enable_compression : bool, optional
        Compress cached schemas to save memory
    enable_fallback : bool, optional
        Fallback to original schema on preprocessing errors
    enable_metrics : bool, optional
        Enable performance metrics collection
    """
    global _CONFIG

    if max_cache_size is not None:
        _CONFIG.max_cache_size = max_cache_size
    if max_recursion_depth is not None:
        _CONFIG.max_recursion_depth = max_recursion_depth
    if max_preprocessing_depth is not None:
        _CONFIG.max_preprocessing_depth = max_preprocessing_depth
    if max_schema_size_bytes is not None:
        _CONFIG.max_schema_size_bytes = max_schema_size_bytes
    if enable_fast_hashing is not None:
        _CONFIG.enable_fast_hashing = enable_fast_hashing
    if enable_compression is not None:
        _CONFIG.enable_compression = enable_compression
    if enable_fallback is not None:
        _CONFIG.enable_fallback = enable_fallback
    if enable_metrics is not None:
        _CONFIG.enable_metrics = enable_metrics


def get_preprocessing_config() -> Dict[str, Any]:
    """Get current preprocessing configuration."""
    return {
        "max_cache_size": _CONFIG.max_cache_size,
        "max_recursion_depth": _CONFIG.max_recursion_depth,
        "max_preprocessing_depth": _CONFIG.max_preprocessing_depth,
        "max_schema_size_bytes": _CONFIG.max_schema_size_bytes,
        "enable_fast_hashing": _CONFIG.enable_fast_hashing,
        "enable_compression": _CONFIG.enable_compression,
        "enable_fallback": _CONFIG.enable_fallback,
        "enable_metrics": _CONFIG.enable_metrics,
    }


def reset_metrics() -> None:
    """Reset all preprocessing metrics."""
    global _METRICS
    _METRICS = PreprocessingMetrics()


def validate_json_schema_basic(schema: Any) -> bool:
    """
    Basic validation that the input looks like a JSON schema.

    This is a lightweight check to catch obviously invalid inputs
    before expensive processing.

    Parameters
    ----------
    schema : Any
        The schema to validate

    Returns
    -------
    bool
        True if schema appears valid, False otherwise
    """
    if not isinstance(schema, dict):
        return False

    # Must have type or be a proper schema object
    if (
        "type" not in schema
        and "$schema" not in schema
        and "anyOf" not in schema
        and "oneOf" not in schema
    ):
        return False

    # If it has a type, it should be valid
    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            valid_types = {
                "string",
                "number",
                "integer",
                "boolean",
                "array",
                "object",
                "null",
            }
            if type_value not in valid_types:
                return False
        elif isinstance(type_value, list):
            valid_types = {
                "string",
                "number",
                "integer",
                "boolean",
                "array",
                "object",
                "null",
            }
            if not all(isinstance(t, str) and t in valid_types for t in type_value):
                return False
            if len(type_value) == 0:  # Empty type array is invalid
                return False
        else:
            return False

    return True
