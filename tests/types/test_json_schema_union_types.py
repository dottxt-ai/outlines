"""Test JSON schema handling with union types (type arrays).

This tests the fix for issue #1383 where dynamic JSON schema creation
fails when optional/union types are nested.
"""

import json
import pytest

from outlines.types import JsonSchema
from outlines.types.dsl import to_regex
from outlines.types.json_schema_utils import preprocess_schema_for_union_types


class TestJsonSchemaUnionTypes:
    """Test cases for JSON schemas with union types."""

    def test_simple_optional_field(self):
        """Test a simple object with an optional field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": ["integer", "null"]},
            },
            "required": ["name"],
        }

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify the type array was converted to anyOf
        assert "anyOf" in parsed["properties"]["age"]
        assert parsed["properties"]["age"]["anyOf"] == [
            {"type": "integer"},
            {"type": "null"},
        ]

        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty

    def test_nested_optional_object(self):
        """Test nested objects with optional types."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": ["object", "null"],
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": ["integer", "null"]},
                    },
                }
            },
        }

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify the outer type array was converted
        assert "anyOf" in parsed["properties"]["person"]
        person_types = parsed["properties"]["person"]["anyOf"]
        assert len(person_types) == 2

        # Verify the nested type array was also converted
        object_type = next(t for t in person_types if t["type"] == "object")
        assert "anyOf" in object_type["properties"]["age"]

        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty

    def test_array_with_optional_items(self):
        """Test arrays with optional item types."""
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": ["string", "null"]}}
            },
        }

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify the items type array was converted
        assert "anyOf" in parsed["properties"]["items"]["items"]
        assert parsed["properties"]["items"]["items"]["anyOf"] == [
            {"type": "string"},
            {"type": "null"},
        ]

        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty

    def test_multiple_types_union(self):
        """Test union with more than two types."""
        schema = {
            "type": "object",
            "properties": {"value": {"type": ["string", "number", "boolean", "null"]}},
        }

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify all types were converted
        assert "anyOf" in parsed["properties"]["value"]
        assert len(parsed["properties"]["value"]["anyOf"]) == 4

        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty

    def test_preserves_type_constraints(self):
        """Test that type-specific constraints are preserved."""
        schema = {
            "type": "object",
            "properties": {
                "text": {
                    "type": ["string", "null"],
                    "minLength": 5,
                    "maxLength": 10,
                    "pattern": "^[A-Z]",
                }
            },
        }

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify constraints were preserved for string type
        string_type = next(
            t for t in parsed["properties"]["text"]["anyOf"] if t["type"] == "string"
        )
        assert string_type["minLength"] == 5
        assert string_type["maxLength"] == 10
        assert string_type["pattern"] == "^[A-Z]"

        # Verify null type has no constraints
        null_type = next(
            t for t in parsed["properties"]["text"]["anyOf"] if t["type"] == "null"
        )
        assert "minLength" not in null_type
        assert "maxLength" not in null_type
        assert "pattern" not in null_type

    def test_no_change_for_single_types(self):
        """Test that single type fields are not modified."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify no anyOf was added
        assert "anyOf" not in parsed["properties"]["name"]
        assert "anyOf" not in parsed["properties"]["age"]
        assert parsed["properties"]["name"]["type"] == "string"
        assert parsed["properties"]["age"]["type"] == "integer"

    def test_deeply_nested_union_types(self):
        """Test schemas with deeply nested structures (10+ levels)."""
        # Create a deeply nested schema with union types at multiple levels
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": ["object", "null"],
                    "properties": {
                        "level2": {
                            "type": ["object", "null"],
                            "properties": {
                                "level3": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "level4": {
                                            "type": ["object", "null"],
                                            "properties": {
                                                "level5": {
                                                    "type": ["object", "null"],
                                                    "properties": {
                                                        "level6": {
                                                            "type": ["object", "null"],
                                                            "properties": {
                                                                "level7": {
                                                                    "type": [
                                                                        "object",
                                                                        "null",
                                                                    ],
                                                                    "properties": {
                                                                        "level8": {
                                                                            "type": [
                                                                                "object",
                                                                                "null",
                                                                            ],
                                                                            "properties": {
                                                                                "level9": {
                                                                                    "type": [
                                                                                        "object",
                                                                                        "null",
                                                                                    ],
                                                                                    "properties": {
                                                                                        "level10": {
                                                                                            "type": [
                                                                                                "string",
                                                                                                "null",
                                                                                            ],
                                                                                            "minLength": 1,
                                                                                        }
                                                                                    },
                                                                                }
                                                                            },
                                                                        }
                                                                    },
                                                                }
                                                            },
                                                        }
                                                    },
                                                }
                                            },
                                        }
                                    },
                                }
                            },
                        }
                    },
                }
            },
        }

        # Test preprocessing doesn't fail on deep nesting
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify the deepest level was processed correctly
        level1 = parsed["properties"]["level1"]["anyOf"]
        object_type = next(t for t in level1 if t["type"] == "object")

        # Navigate down to level10 to verify conversion
        current = object_type
        for i in range(2, 11):
            level_name = f"level{i}"
            if i < 10:
                current = next(
                    t
                    for t in current["properties"][level_name]["anyOf"]
                    if t["type"] == "object"
                )
            else:
                # level10 should have string/null union
                level10_union = current["properties"][level_name]["anyOf"]
                assert len(level10_union) == 2
                string_type = next(t for t in level10_union if t["type"] == "string")
                assert string_type["minLength"] == 1

        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty

    def test_interaction_with_existing_anyof(self):
        """Test schemas that already contain anyOf/oneOf constructs."""
        schema = {
            "type": "object",
            "properties": {
                "mixed_field": {
                    "anyOf": [
                        {"type": "string", "minLength": 5},
                        {
                            "type": "object",
                            "properties": {"nested": {"type": ["integer", "null"]}},
                        },
                    ]
                },
                "oneof_field": {
                    "oneOf": [{"type": ["string", "null"]}, {"type": "number"}]
                },
            },
        }

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify existing anyOf is preserved and nested unions are converted
        mixed_anyof = parsed["properties"]["mixed_field"]["anyOf"]
        assert len(mixed_anyof) == 2

        # Check that nested union was converted in the object alternative
        object_alt = next(alt for alt in mixed_anyof if alt["type"] == "object")
        nested_union = object_alt["properties"]["nested"]["anyOf"]
        assert len(nested_union) == 2
        assert {"type": "integer"} in nested_union
        assert {"type": "null"} in nested_union

        # Verify oneOf field union was converted
        oneof_alternatives = parsed["properties"]["oneof_field"]["oneOf"]
        string_null_alt = next(alt for alt in oneof_alternatives if "anyOf" in alt)
        assert len(string_null_alt["anyOf"]) == 2

        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty

    def test_large_schema_performance(self):
        """Test schemas with many properties to ensure performance is acceptable."""
        # Create a schema with 100 properties, each with union types
        properties = {}
        for i in range(100):
            properties[f"field_{i}"] = {
                "type": ["string", "integer", "null"],
                "minLength": 1 if i % 3 == 0 else None,
                "minimum": 0 if i % 3 == 1 else None,
            }
            # Remove None values
            properties[f"field_{i}"] = {
                k: v for k, v in properties[f"field_{i}"].items() if v is not None
            }

        schema = {"type": "object", "properties": properties}

        import time

        start_time = time.time()

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        processing_time = time.time() - start_time

        # Verify all fields were processed
        for i in range(100):
            field_name = f"field_{i}"
            assert "anyOf" in parsed["properties"][field_name]
            assert len(parsed["properties"][field_name]["anyOf"]) == 3

        # Performance should be reasonable (less than 5 seconds for 100 fields)
        assert (
            processing_time < 5.0
        ), f"Processing took {processing_time:.2f}s, which is too slow"

        # Test that JsonSchema can handle it (basic instantiation check)
        JsonSchema(schema)
        # Note: We don't test regex generation here as it would be very slow

    def test_complex_array_scenarios(self):
        """Test complex array scenarios with nested union types."""
        schema = {
            "type": "object",
            "properties": {
                "array_of_unions": {
                    "type": "array",
                    "items": {
                        "type": ["object", "string", "null"],
                        "properties": {
                            "nested_array": {
                                "type": "array",
                                "items": {"type": ["number", "boolean"]},
                            }
                        },
                    },
                },
                "multi_dimensional": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": ["string", "null"]}},
                },
            },
        }

        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)

        # Verify array_of_unions was processed
        array_items = parsed["properties"]["array_of_unions"]["items"]["anyOf"]
        assert len(array_items) == 3

        # Find the object alternative and check nested array
        object_alt = next(alt for alt in array_items if alt["type"] == "object")
        nested_array_items = object_alt["properties"]["nested_array"]["items"]["anyOf"]
        assert len(nested_array_items) == 2
        assert {"type": "number"} in nested_array_items
        assert {"type": "boolean"} in nested_array_items

        # Verify multi_dimensional was processed
        multi_dim_items = parsed["properties"]["multi_dimensional"]["items"]["items"][
            "anyOf"
        ]
        assert len(multi_dim_items) == 2
        assert {"type": "string"} in multi_dim_items
        assert {"type": "null"} in multi_dim_items

        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty

    def test_invalid_schemas_handling(self):
        """Test handling of invalid or malformed schemas."""
        # Test invalid JSON
        with pytest.raises(ValueError, match="Invalid JSON schema"):
            preprocess_schema_for_union_types("{invalid json}")

        # Test empty type array (edge case)
        schema_empty_type = {"type": "object", "properties": {"field": {"type": []}}}

        # Should not crash, but will produce empty anyOf
        preprocessed = preprocess_schema_for_union_types(schema_empty_type)
        parsed = json.loads(preprocessed)
        assert parsed["properties"]["field"]["anyOf"] == []

    def test_caching_mechanism(self):
        """Test that the caching mechanism works correctly."""
        from outlines.types.json_schema_utils import clear_schema_cache, get_cache_stats

        # Clear cache to start fresh
        clear_schema_cache()
        initial_stats = get_cache_stats()
        assert initial_stats["cache_size"] == 0

        schema = {
            "type": "object",
            "properties": {"field": {"type": ["string", "null"]}},
        }

        # First call should add to cache
        result1 = preprocess_schema_for_union_types(schema)
        stats_after_first = get_cache_stats()
        assert stats_after_first["cache_size"] == 1

        # Second call with same schema should use cache
        result2 = preprocess_schema_for_union_types(schema)
        stats_after_second = get_cache_stats()
        assert stats_after_second["cache_size"] == 1  # No new cache entry
        assert result1 == result2  # Results should be identical

        # Different schema should add new cache entry
        different_schema = {
            "type": "object",
            "properties": {"other_field": {"type": ["integer", "null"]}},
        }

        result3 = preprocess_schema_for_union_types(different_schema)
        stats_after_third = get_cache_stats()
        assert stats_after_third["cache_size"] == 2
        assert result3 != result1  # Different schemas produce different results

    def test_optimization_skip_no_unions(self):
        """Test that schemas without union types are returned unchanged (optimization)."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "nested": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                },
            },
        }

        original_json = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        result = preprocess_schema_for_union_types(schema)

        # Result should be identical to input since no processing was needed
        assert result == original_json

        # Parse to verify structure is unchanged
        parsed = json.loads(result)
        assert parsed == schema

    def test_thread_safety_concurrent_access(self):
        """Test that concurrent access to the cache is thread-safe."""
        import threading
        import time
        from outlines.types.json_schema_utils import clear_schema_cache, get_cache_stats

        # Clear cache to start fresh
        clear_schema_cache()

        results = {}
        errors = []

        def worker(thread_id: int):
            """Worker function that processes schemas concurrently."""
            try:
                for i in range(10):
                    schema = {
                        "type": "object",
                        "properties": {
                            f"field_{thread_id}_{i}": {"type": ["string", "null"]}
                        },
                    }
                    result = preprocess_schema_for_union_types(schema)
                    results[f"{thread_id}_{i}"] = result
                    time.sleep(
                        0.001
                    )  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Verify all results are valid
        assert len(results) == 50  # 5 threads * 10 schemas each
        for key, result in results.items():
            parsed = json.loads(result)
            assert "anyOf" in parsed["properties"][f"field_{key}"]

        # Verify cache statistics
        stats = get_cache_stats()
        assert stats["cache_size"] > 0
        assert stats["cache_size"] <= stats["max_cache_size"]

    def test_max_depth_protection(self):
        """Test that very deep schemas don't cause stack overflow."""
        # Create a deeply nested schema (beyond the 50-level limit)
        deep_schema = {"type": "object"}
        current = deep_schema

        # Create 60 levels of nesting (beyond max_depth=50)
        for i in range(60):
            current["properties"] = {
                f"level_{i}": {
                    "type": (
                        "object" if i < 59 else ["string", "null"]
                    )  # Only the deepest has union
                }
            }
            current = current["properties"][f"level_{i}"]

        # This should not crash and should not find the deep type array
        # due to depth limiting at 50 levels
        from outlines.types.json_schema_utils import _contains_type_arrays

        # Test with low depth limit - should return False due to depth limiting
        result = _contains_type_arrays(deep_schema, max_depth=10)
        assert result is False, "Should not find type arrays due to depth limiting"

        # Test with sufficient depth limit - should find the type array
        # Note: 60 levels of nesting creates depth ~120, so we need higher limit
        result = _contains_type_arrays(deep_schema, max_depth=150)
        assert (
            result is True
        ), "Should find the deep type array with sufficient depth limit"

        # Test that the function doesn't crash on very deep structures
        very_deep_schema = {"type": "object"}
        current = very_deep_schema
        for i in range(200):  # Create extremely deep nesting
            current["properties"] = {f"level_{i}": {"type": "object"}}
            current = current["properties"][f"level_{i}"]

        # Add type array at the very end (depth 200)
        current["type"] = ["string", "null"]

        # This should not crash even with very deep structure
        result = _contains_type_arrays(very_deep_schema, max_depth=50)
        assert result is False, "Should handle very deep structures without crashing"

    def test_early_termination_performance(self):
        """Test that type array detection terminates early for better performance."""
        import time

        # Create a large schema with type array at the beginning
        schema_early_match = {
            "type": "object",
            "properties": {
                "first_field": {"type": ["string", "null"]}  # Type array here
            },
        }

        # Add many more properties after the first one
        for i in range(1000):
            schema_early_match["properties"][f"field_{i}"] = {"type": "string"}

        # Create a similar schema with type array at the end
        schema_late_match = {"type": "object", "properties": {}}

        # Add many properties without type arrays
        for i in range(1000):
            schema_late_match["properties"][f"field_{i}"] = {"type": "string"}

        # Add type array at the end
        schema_late_match["properties"]["last_field"] = {"type": ["string", "null"]}

        from outlines.types.json_schema_utils import _contains_type_arrays

        # Time the early match case
        start_time = time.time()
        result_early = _contains_type_arrays(schema_early_match)
        early_time = time.time() - start_time

        # Time the late match case
        start_time = time.time()
        result_late = _contains_type_arrays(schema_late_match)
        late_time = time.time() - start_time

        # Both should find type arrays
        assert result_early is True
        assert result_late is True

        # Early termination should be significantly faster
        # Allow some variance but early should be at least 2x faster
        assert (
            early_time * 2 < late_time
        ), f"Early termination not working: early={early_time:.4f}s, late={late_time:.4f}s"

    def test_circular_reference_detection(self):
        """Test that circular references are properly detected and handled."""
        from outlines.types.json_schema_utils import (
            _contains_type_arrays,
            _preprocess_schema_dict_safe,
        )

        # Create a schema with circular reference
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "child": None,  # Will be set to create circular reference
            },
        }

        # Create circular reference - child points back to the schema
        schema["properties"]["child"] = schema

        # Test that _contains_type_arrays handles circular references gracefully
        result = _contains_type_arrays(schema)
        assert result is False  # No type arrays in this circular schema

        # Add a type array to test circular reference with type arrays
        circular_schema_with_union = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},  # This has a type array
                "child": None,
            },
        }
        circular_schema_with_union["properties"]["child"] = circular_schema_with_union

        # Should detect the type array despite circular reference
        result = _contains_type_arrays(circular_schema_with_union)
        assert result is True

        # Test that preprocessing detects and reports circular references
        with pytest.raises(ValueError, match="Circular reference detected"):
            _preprocess_schema_dict_safe(circular_schema_with_union)

    def test_semantic_validation_anyof_equivalence(self):
        """Test that anyOf conversion preserves semantic equivalence."""
        # Test cases that should be semantically equivalent
        test_cases = [
            # Simple optional field
            {
                "original": {"type": ["string", "null"]},
                "expected_types": ["string", "null"],
            },
            # Optional with constraints
            {
                "original": {"type": ["string", "null"], "minLength": 5},
                "expected_types": ["string", "null"],
                "string_constraints": {"minLength": 5},
            },
            # Multiple types with constraints
            {
                "original": {
                    "type": ["integer", "string", "null"],
                    "minimum": 0,
                    "minLength": 1,
                },
                "expected_types": ["integer", "string", "null"],
                "integer_constraints": {"minimum": 0},
                "string_constraints": {"minLength": 1},
            },
        ]

        for i, case in enumerate(test_cases):
            original = case["original"]

            # Convert using our function
            preprocessed = preprocess_schema_for_union_types(original)
            converted = json.loads(preprocessed)

            # Verify structure
            assert "anyOf" in converted, f"Case {i}: anyOf missing in converted schema"
            anyof_alternatives = converted["anyOf"]

            # Check that all expected types are present
            actual_types = [alt["type"] for alt in anyof_alternatives]
            expected_types = case["expected_types"]
            assert sorted(actual_types) == sorted(
                expected_types
            ), f"Case {i}: Types don't match"

            # Verify constraints are correctly distributed
            if "string_constraints" in case:
                string_alt = next(
                    alt for alt in anyof_alternatives if alt["type"] == "string"
                )
                for key, value in case["string_constraints"].items():
                    assert (
                        string_alt[key] == value
                    ), f"Case {i}: String constraint {key} mismatch"

            if "integer_constraints" in case:
                integer_alt = next(
                    alt for alt in anyof_alternatives if alt["type"] == "integer"
                )
                for key, value in case["integer_constraints"].items():
                    assert (
                        integer_alt[key] == value
                    ), f"Case {i}: Integer constraint {key} mismatch"

    def test_configuration_system(self):
        """Test the configuration system for preprocessing behavior."""
        from outlines.types.json_schema_utils import (
            configure_preprocessing,
            get_preprocessing_config,
            clear_schema_cache,
            get_cache_stats,
        )

        # Get original config
        original_config = get_preprocessing_config()

        try:
            # Test configuration changes
            configure_preprocessing(
                max_cache_size=50,
                max_recursion_depth=25,
                enable_fast_hashing=False,
                enable_compression=False,
                enable_fallback=False,
                enable_metrics=True,
            )

            new_config = get_preprocessing_config()
            assert new_config["max_cache_size"] == 50
            assert new_config["max_recursion_depth"] == 25
            assert new_config["enable_fast_hashing"] is False
            assert new_config["enable_compression"] is False
            assert new_config["enable_fallback"] is False
            assert new_config["enable_metrics"] is True

            # Test that cache size is respected
            clear_schema_cache()

            # Add schemas to fill cache beyond new limit
            for i in range(60):
                schema = {"type": ["string", "null"], "minLength": i}
                preprocess_schema_for_union_types(schema)

            stats = get_cache_stats()
            assert stats["cache_size"] <= 50  # Should respect new limit

        finally:
            # Restore original configuration
            configure_preprocessing(**original_config)

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        from outlines.types.json_schema_utils import (
            configure_preprocessing,
            reset_metrics,
            get_cache_stats,
            clear_schema_cache,
        )

        # Enable metrics and reset
        configure_preprocessing(enable_metrics=True)
        reset_metrics()
        clear_schema_cache()

        # Perform some operations
        schema1 = {"type": ["string", "null"]}
        schema2 = {"type": ["integer", "null"]}

        # First calls (cache misses)
        preprocess_schema_for_union_types(schema1)
        preprocess_schema_for_union_types(schema2)

        # Second calls (cache hits)
        preprocess_schema_for_union_types(schema1)
        preprocess_schema_for_union_types(schema2)

        stats = get_cache_stats()
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert "preprocessing_count" in stats
        assert "total_processing_time" in stats
        assert "avg_processing_time" in stats

        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 2
        assert stats["cache_hit_rate"] == 0.5
        assert stats["preprocessing_count"] == 2  # Only first calls do preprocessing
        assert stats["total_processing_time"] > 0

    def test_dos_protection(self):
        """Test DoS protection against very large schemas."""
        from outlines.types.json_schema_utils import configure_preprocessing

        # Configure small size limit for testing
        configure_preprocessing(max_schema_size_bytes=1000)

        try:
            # Create a very large schema
            large_schema = {"type": "object", "properties": {}}
            for i in range(1000):  # Make it large enough to exceed limit
                large_schema["properties"][f"field_{i}"] = {"type": ["string", "null"]}

            # Should raise error due to size limit
            with pytest.raises(ValueError, match="Schema size exceeds maximum"):
                preprocess_schema_for_union_types(large_schema)

        finally:
            # Restore reasonable size limit
            configure_preprocessing(max_schema_size_bytes=10 * 1024 * 1024)

    def test_graceful_fallback(self):
        """Test graceful fallback when preprocessing fails."""
        from outlines.types.json_schema_utils import (
            configure_preprocessing,
            get_cache_stats,
        )

        # Enable fallback mode
        configure_preprocessing(enable_fallback=True, enable_metrics=True)

        # Create a schema that will cause preprocessing to fail
        # by setting very low depth limit
        configure_preprocessing(max_preprocessing_depth=1)

        try:
            schema = {
                "type": "object",
                "properties": {
                    "level1": {
                        "type": "object",
                        "properties": {
                            "level2": {
                                "type": ["string", "null"]
                            }  # Deep enough to exceed limit
                        },
                    }
                },
            }

            # Should fallback to original schema without raising error
            result = preprocess_schema_for_union_types(schema)
            original_json = json.dumps(schema, sort_keys=True, separators=(",", ":"))
            assert result == original_json

            # Check that fallback was recorded in metrics
            stats = get_cache_stats()
            assert stats["fallback_count"] > 0

        finally:
            # Restore normal depth limit
            configure_preprocessing(max_preprocessing_depth=100, enable_fallback=True)

    def test_input_validation(self):
        """Test input validation for obviously invalid schemas."""
        # Test various invalid inputs
        invalid_schemas = [
            {},  # Empty object
            {"invalid": "schema"},  # No type or schema indicators
            {"type": "invalid_type"},  # Invalid type
            {"type": []},  # Empty type array
            {"type": ["string", "invalid_type"]},  # Mixed valid/invalid types
            {"type": 123},  # Non-string, non-array type
        ]

        for invalid_schema in invalid_schemas:
            with pytest.raises(
                ValueError, match="not appear to be a valid JSON schema"
            ):
                preprocess_schema_for_union_types(invalid_schema)

        # Test that valid schemas still work
        valid_schemas = [
            {"type": "string"},
            {"type": ["string", "null"]},
            {"$schema": "http://json-schema.org/draft-07/schema#"},
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        ]

        for valid_schema in valid_schemas:
            # Should not raise error
            result = preprocess_schema_for_union_types(valid_schema)
            assert isinstance(result, str)

    def test_compression_and_fast_hashing(self):
        """Test compression and fast hashing features."""
        from outlines.types.json_schema_utils import (
            configure_preprocessing,
            clear_schema_cache,
            get_cache_stats,
        )

        # Test with compression enabled
        configure_preprocessing(enable_compression=True, enable_fast_hashing=True)
        clear_schema_cache()

        schema = {
            "type": ["string", "null"],
            "description": "A" * 1000,
        }  # Large description
        preprocess_schema_for_union_types(schema)

        stats_compressed = get_cache_stats()

        # Test with compression disabled
        configure_preprocessing(enable_compression=False)
        clear_schema_cache()

        preprocess_schema_for_union_types(schema)

        stats_uncompressed = get_cache_stats()

        # Compressed should use less memory (though this test might be flaky
        # if the schema is too small to compress effectively)
        # We mainly test that both modes work without errors
        assert stats_compressed["cache_size"] == 1
        assert stats_uncompressed["cache_size"] == 1
        assert "total_cache_memory_bytes" in stats_compressed
        assert "total_cache_memory_bytes" in stats_uncompressed
