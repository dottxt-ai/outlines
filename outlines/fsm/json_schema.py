import dataclasses
import inspect
import itertools
import json
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import yaml
from jsonschema.protocols import Validator
from pydantic import BaseModel, create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012

# allow `\"`, `\\`, or any character which isn't a control sequence
STRING_INNER = r'([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])'
STRING = f'"{STRING_INNER}*"'

INTEGER = r"(-)?(0|[1-9][0-9]*)"
NUMBER = rf"({INTEGER})(\.[0-9]+)?([eE][+-][0-9]+)?"
BOOLEAN = r"(true|false)"
NULL = r"null"
WHITESPACE = r"[ ]?"

DATE_TIME = r'"(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{3})?(Z)?"'
DATE = r'"(?:\d{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])"'
TIME = r'"(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\\.[0-9]+)?(Z)?"'
UUID = r'"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"'


def load_yaml(yaml_str: str) -> Any:
    """Parse a YAML string and return the corresponding Python object."""
    return yaml.safe_load(yaml_str)


def build_regex_from_schema(
    schema: str,
    whitespace_pattern: Optional[str] = None,
    mode: str = "json",
    strict_json_schema_subset: bool = True,
):
    """Turn a JSON schema into a regex that matches any JSON object that follows
    this schema.

    JSON Schema is a declarative language that allows to annotate JSON documents
    with types and descriptions. These schemas can be generated from any Python
    datastructure that has type annotation: namedtuples, dataclasses, Pydantic
    models. And by ensuring that the generation respects the schema we ensure
    that the output can be parsed into these objects.
    This function parses the provided schema and builds a generation schedule which
    mixes deterministic generation (fixed strings), and sampling with constraints.

    Parameters
    ----------
    schema
        A string that represents a JSON Schema.
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`
    mode
        Either `json` or `yaml`, determines the structure of the generated output
    strict_json_schema_subset
        For `items` and `properties`, the JSON Schema spec by default allows these to be unconstrained
        if not set. This is usually undesired behavior, so by default strict_json_schema_subset is True.


    Returns
    -------
    A generation schedule. A list of strings that represent the JSON
    schema's structure and regular expression that define the structure of
    the fields.

    References
    ----------
    .. [0] JSON Schema. https://json-schema.org/

    """

    schema = json.loads(schema)
    Validator.check_schema(schema)

    # Build reference resolver
    schema = Resource(contents=schema, specification=DRAFT202012)
    uri = schema.id() if schema.id() is not None else ""
    registry = Registry().with_resource(uri=uri, resource=schema)
    resolver = registry.resolver()

    content = schema.contents

    if mode == "json":
        return JSONSchemaRegexGenerator(
            resolver,
            whitespace_pattern,
            strict_json_schema_subset=strict_json_schema_subset,
        ).to_regex(content)
    elif mode == "yaml":
        return YAMLRegexGenerator(
            resolver,
            whitespace_pattern,
            strict_json_schema_subset=strict_json_schema_subset,
        ).to_regex(content)
    else:
        raise ValueError(f"invalid mode: {mode}")


def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
    """Convert a JSON schema to a string.

    Parameters
    ----------
    json_schema
        The JSON schema.

    Returns
    -------
    str
        The JSON schema converted to a string.

    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def validate_quantifiers(
    min_bound: Optional[str], max_bound: Optional[str], start_offset: int = 0
) -> Tuple[str, str]:
    """
    Ensures that the bounds of a number are valid. Bounds are used as quantifiers in the regex.

    Parameters
    ----------
    min_bound
        The minimum value that the number can take.
    max_bound
        The maximum value that the number can take.
    start_offset
        Number of elements that are already present in the regex but still need to be counted.
        ex: if the regex is already "(-)?(0|[1-9][0-9])", we will always have at least 1 digit, so the start_offset is 1.

    Returns
    -------
    min_bound
        The minimum value that the number can take.
    max_bound
        The maximum value that the number can take.

    Raises
    ------
    ValueError
        If the minimum bound is greater than the maximum bound.

    TypeError or ValueError
        If the minimum bound is not an integer or None.
        or
        If the maximum bound is not an integer or None.
    """
    min_bound = "" if min_bound is None else str(int(min_bound) - start_offset)
    max_bound = "" if max_bound is None else str(int(max_bound) - start_offset)
    if min_bound and max_bound:
        if int(max_bound) < int(min_bound):
            raise ValueError("max bound must be greater than or equal to min bound")
    return min_bound, max_bound


def get_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    signature = inspect.signature(fn)
    arguments = {}
    for name, arg in signature.parameters.items():
        if arg.annotation == inspect._empty:
            raise ValueError("Each argument must have a type annotation")
        else:
            arguments[name] = (arg.annotation, ...)

    try:
        fn_name = fn.__name__
    except Exception as e:
        fn_name = "Arguments"
        warnings.warn(
            f"The function name could not be determined. Using default name 'Arguments' instead. For debugging, here is exact error:\n{e}",
            category=UserWarning,
        )
    model = create_model(fn_name, **arguments)

    return model.model_json_schema()


@dataclasses.dataclass
class Context:
    """Context for json schema rule application"""

    recursion_depth: int = 0
    nesting_level: int = 0

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def increment(self, attr: str, value: int = 1) -> "Context":
        """Return a **new** Context with the specified attribute incremented by `value`"""
        return dataclasses.replace(self, **{attr: getattr(self, attr) + value})

    def __repr__(self):
        return f"Context({self.__dict__})"


class JSONSchemaRegexGenerator:
    """Translate a JSON Schema instance into a regex that validates the schema.

    Note
    ----
    Many features of JSON schema are missing:
    - Handle constraints on numbers
    - Handle special patterns: `date`, `uri`, etc.

    This does not support recursive definitions.

    Parameters
    ----------
    resolver
        An object that resolves references to other instances within a schema
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`
    recursion_level
        For unconstrained objects and lists ond many levels deep the pattern should be constructed.
    strict_json_schema_subset
        For `items` and `properties`, the JSON Schema spec by default allows these to be unconstrained
        if not set. This is usually undesired behavior, so by default strict_json_schema_subset is True.
    """

    # Never impacted by parameters
    STATIC_PRIMATIVES = {"boolean", "null"}
    # Default value of primatives (when provided no parameters)
    FORMAT_PRIMATIVE = {
        "null": NULL,
        "boolean": BOOLEAN,
        "number": NUMBER,
        "integer": INTEGER,
        "string": STRING,
    }
    FORMAT_STRING = {
        "uuid": UUID,
        "date-time": DATE_TIME,
        "date": DATE,
        "time": TIME,
    }

    def __init__(
        self,
        resolver: Resolver,
        whitespace_pattern: Optional[str] = None,
        max_nesting_level: int = 2,
        strict_json_schema_subset: bool = True,
    ):
        self.resolver = resolver
        self.ws = WHITESPACE if whitespace_pattern is None else whitespace_pattern
        self.max_nesting_level = max_nesting_level
        self.strict_json_schema_subset = strict_json_schema_subset

    def _validate_node(self, node: Any, ctx: Context):
        """Validate the JSON Schema node for unsupported features and recursion limits."""
        if ctx.recursion_depth > 256:
            raise NotImplementedError(
                "Recursive schemas aren't currently available with Outlines."
            )

        if node is True:
            return

        if node is False:
            # this should be implemented along-side `visit_not(...)`
            raise NotImplementedError("schema = False isn't available with Outlines.")

        # keys have no handling
        not_implemented_keys = [
            "dependentSchemas",
            "unevaluatedItems",
            "unevaluatedProperties",
            "contains",
            "patternProperties",
            "maximum",
            "default",
            "__proto__",
        ]
        # keys coinciding within same object not handled
        not_implemented_key_pairs = [
            ("allOf", "anyOf"),
            ("properties", "anyOf"),
        ]

        node_invalid_keys = set(node) & set(not_implemented_keys)
        if node_invalid_keys:
            raise NotImplementedError(
                f"Cannot handle the keys: {node_invalid_keys}. Please open an Outlines issue."
            )
        for k in not_implemented_key_pairs:
            if not (set(k) - set(node.keys())):
                raise NotImplementedError(
                    f"Cannot simultaneously use the keys: {k}. Please open an Outlines issue."
                )

    def to_regex(self, node: Any, ctx: Optional[Context] = None):
        """Convert a JSON Schema node into a regular expression pattern."""
        ctx = (
            ctx.increment("recursion_depth")
            if ctx
            else Context(nesting_level=0, recursion_depth=0)
        )
        self._validate_node(node, ctx)

        # Handle unconstrained nodes
        if node in ({}, True):
            return self.visit_unconstrained({}, ctx)

        # Handle multiple types (via anyOf)
        if isinstance(node.get("type"), list):
            subpatterns = [self.to_regex({"type": t}, ctx) for t in node["type"]]
            return self.format_anyOf(subpatterns)

        # Visit based on node attributes
        node_attr_to_visitor = {
            "$ref": self.visit_ref,
            "allOf": self.visit_allOf,
            "anyOf": self.visit_anyOf,
            "oneOf": self.visit_oneOf,
            "enum": self.visit_enum,
            "prefixItems": self.visit_array,
            "items": self.visit_array,
            "properties": self.visit_object,
            "const": self.visit_string,
            "pattern": self.visit_string,
        }
        for attr, visitor in node_attr_to_visitor.items():
            if attr in node:
                return (
                    f"({visitor(node, ctx)})"
                    if ctx.nesting_level > 0
                    else visitor(node, ctx)
                )

        # Visit based on type
        type_to_visitor = {
            "number": self.visit_number,
            "integer": self.visit_integer,
            "string": self.visit_string,
            "object": self.visit_object,
            "array": self.visit_array,
        }
        if node.get("type") in self.STATIC_PRIMATIVES:
            return self.FORMAT_PRIMATIVE[node["type"]]
        if node.get("type") in type_to_visitor:
            return type_to_visitor[node["type"]](node, ctx)

        return self.visit_notimplemented(node, ctx)

    ##########
    # VISITORS
    ##########
    def visit_ref(self, node: Any, ctx: Context):
        path = node["$ref"]
        if path == "#":
            raise NotImplementedError("Recursive schemas aren't supported")
        new_node = self.resolver.lookup(path).contents
        return self.to_regex(new_node, ctx)

    def visit_object(self, node: Any, ctx: Context):
        """
        Handle JSON Schema `object` rules

        additionalProperties handling:
            pattern for json object with values defined by instance["additionalProperties"]
            enforces value type constraints recursively, "minProperties", and "maxProperties"
            doesn't enforce "required", "dependencies", "propertyNames" "any/all/on Of"

        TODO: the json-schema compliant implementation is as follows:
        - properties and additionalProperties can both be set simultaneously
        - min/maxProperties can be specified even if properties has constraints set
        """
        value_ctx = ctx.increment("nesting_level")

        # TODO: handling for node["unevaluatedProperties"]
        properties = node.get("properties", not self.strict_json_schema_subset)
        properties = {} if properties is True else properties
        required_properties = node.get("required", [])
        additional_properties = node.get("additionalProperties")

        if properties and additional_properties:
            raise NotImplementedError(
                "`properties` & `additionalProperties != False` not implemented. Please open an Outlines issue."
            )

        elif properties and "minProperties" in node or "maxProperties" in node:
            raise NotImplementedError(
                "properties and minProperties / maxProperties not implemented. Please open an Outlines issue."
            )

        elif properties:
            property_details = [
                {
                    "key_pattern": self.format_literal(name),
                    "value_pattern": self.to_regex(value, value_ctx),
                    "is_required": name in required_properties,
                }
                for name, value in properties.items()
            ]
            if any(pd["is_required"] for pd in property_details):
                return self.format_object_with_required_properties(
                    property_details, ctx
                )
            else:
                return self.format_object_properties_all_optional(property_details, ctx)

        elif additional_properties is False:
            return self.format_empty_object()

        else:
            if additional_properties in (True, None):
                value_pattern = self.visit_unconstrained(node, value_ctx)
            else:
                # Object with arbitrary key name, constrained value
                value_pattern = self.to_regex(additional_properties, value_ctx)

            return self.format_object_with_additional_properties(
                value_pattern,
                ctx,
                min_properties=node.get("minProperties"),
                max_properties=node.get("maxProperties"),
            )

    def visit_array(self, node: Any, ctx: Context):
        """Handle JSON Schema `array` rules with optional item constraints."""

        # TODO: handling for node["unevaluatedItems"]
        # TODO: handling for node["additionalItems"]
        # TODO: handling for node["uniqueItems"]
        if "uniqueItems" in node:
            raise NotImplementedError(
                "uniqueItems is not implemented. Please open an Outlines issue."
            )

        elem_ctx = ctx.increment("nesting_level")

        items = node.get("items", not self.strict_json_schema_subset)

        if node.get("prefixItems") is not None:
            # `prefixItems` determines types at each idx, which precedes `items` rules
            if items in (True, None):
                suffix_elem_pattern = self.visit_unconstrained(node, elem_ctx)
            elif items is False:
                suffix_elem_pattern = None
            else:
                suffix_elem_pattern = self.to_regex(items, elem_ctx)

            prefix_subpatterns = [
                self.to_regex(item, elem_ctx) for item in node["prefixItems"]
            ]
            return self.format_prefixItems(prefix_subpatterns, ctx, suffix_elem_pattern)

        else:
            # handle simple case: no prefix items
            if node.get("items") in (True, None):  # noqa
                items_regex = self.visit_unconstrained(node, elem_ctx)
            else:
                items_regex = self.to_regex(node["items"], elem_ctx)
            return self.format_array(
                items_regex, ctx, node.get("minItems"), node.get("maxItems")
            )

    def visit_number(self, node: Any, ctx: Context):
        quantifier_keys = [
            "minDigitsInteger",
            "maxDigitsInteger",
            "minDigitsFraction",
            "maxDigitsFraction",
            "minDigitsExponent",
            "maxDigitsExponent",
        ]
        if any([qk in node for qk in quantifier_keys]):
            min_digits_integer, max_digits_integer = validate_quantifiers(
                node.get("minDigitsInteger"),
                node.get("maxDigitsInteger"),
                start_offset=1,
            )
            min_digits_fraction, max_digits_fraction = validate_quantifiers(
                node.get("minDigitsFraction"), node.get("maxDigitsFraction")
            )
            min_digits_exponent, max_digits_exponent = validate_quantifiers(
                node.get("minDigitsExponent"), node.get("maxDigitsExponent")
            )
            return self.format_number_range(
                min_digits_integer,
                max_digits_integer,
                min_digits_fraction,
                max_digits_fraction,
                min_digits_exponent,
                max_digits_exponent,
            )
        else:
            return self.FORMAT_PRIMATIVE["number"]

    def visit_integer(self, node: Any, ctx: Context):
        if "maxDigits" in node or "minDigits" in node:
            min_digits, max_digits = validate_quantifiers(
                node.get("minDigits"), node.get("maxDigits"), start_offset=1
            )
            return self.format_integer_range(min_digits, max_digits)
        else:
            return self.FORMAT_PRIMATIVE["integer"]

    def visit_string(self, node: Any, ctx: Context):
        if "const" in node:
            return self.format_literal(node["const"])
        if "maxLength" in node or "minLength" in node:
            min_length, max_length = validate_quantifiers(
                node.get("minLength"), node.get("maxLength")
            )
            return self.format_string_length(min_length, max_length)
        elif "pattern" in node:
            return self.format_string_pattern(node["pattern"])
        elif "format" in node:
            return self.format_string_format(node["format"])
        return self.FORMAT_PRIMATIVE["string"]

    def visit_enum(self, node: Any, ctx: Context):
        """
        The enum keyword is used to restrict a value to a fixed set of values. It
        must be an array with at least one element, where each element is unique.
        """
        choices = [self.format_literal(choice) for choice in node["enum"]]
        return self.format_anyOf(choices)

    def visit_allOf(self, node: Any, ctx: Context):
        subpatterns = [self.to_regex(subschema, ctx) for subschema in node["allOf"]]
        return self.format_allOf(subpatterns)

    def visit_anyOf(self, node: Any, ctx: Context):
        subpatterns = [self.to_regex(subschema, ctx) for subschema in node["anyOf"]]
        return self.format_anyOf(subpatterns)

    def visit_oneOf(self, node: Any, ctx: Context):
        subpatterns = [self.to_regex(subschema, ctx) for subschema in node["oneOf"]]
        return self.format_oneOf(subpatterns)

    def visit_notimplemented(self, node: Any, ctx: Context):
        raise NotImplementedError(
            f"Handler for node `{node}` is not implemented. Please open an Outlines issue."
        )

    ############
    # FORMATTERS
    ############
    def format_number_range(
        self,
        min_digits_integer,
        max_digits_integer,
        min_digits_fraction,
        max_digits_fraction,
        min_digits_exponent,
        max_digits_exponent,
    ):
        integers_quantifier = (
            f"{{{min_digits_integer},{max_digits_integer}}}"
            if min_digits_integer or max_digits_integer
            else "*"
        )
        fraction_quantifier = (
            f"{{{min_digits_fraction},{max_digits_fraction}}}"
            if min_digits_fraction or max_digits_fraction
            else "+"
        )
        exponent_quantifier = (
            f"{{{min_digits_exponent},{max_digits_exponent}}}"
            if min_digits_exponent or max_digits_exponent
            else "+"
        )
        return rf"((-)?(0|[1-9][0-9]{integers_quantifier}))(\.[0-9]{fraction_quantifier})?([eE][+-][0-9]{exponent_quantifier})?"

    def format_integer_range(self, min_digits=None, max_digits=None):
        if min_digits or max_digits:
            num_items_pattern = f"{{{min_digits},{max_digits}}}"
        else:
            num_items_pattern = "*"

        return rf"(-)?(0|[1-9][0-9]{num_items_pattern})"

    def format_string_length(self, min_length, max_length):
        return f'"{STRING_INNER}{{{min_length},{max_length}}}"'

    def format_string_pattern(self, pattern: str):
        if pattern[0] == "^" and pattern[-1] == "$":
            pattern_string_inner = pattern[1:-1]
        else:
            pattern_string_inner = pattern
        return f'"{pattern_string_inner}"'

    def format_string_format(self, fmt: str):
        format_regex = self.FORMAT_STRING.get(fmt)
        if format_regex:
            return format_regex
        raise NotImplementedError(
            f"Format {fmt} is not supported. Please open an Outlines issue."
        )

    def format_property_kv(
        self, key_pattern: str, value_pattern: str, ctx: Context
    ) -> str:
        return f"{self.ws}{key_pattern}{self.ws}(:){self.ws}{value_pattern}"

    def format_empty_object(self):
        return r"\{" + self.ws + r"\}"

    def format_object_properties_all_optional(
        self, property_details: List[Dict], ctx: Context
    ):
        property_subregexes = [
            self.format_property_kv(pd["key_pattern"], pd["value_pattern"], ctx)
            for pd in property_details
        ]
        possible_patterns = [
            f"{self.ws},".join(combination)
            for i in range(1, len(property_subregexes) + 1)
            for combination in itertools.combinations(property_subregexes, i)
        ]
        inner = f"({'|'.join(possible_patterns)})?"
        return r"\{" + inner + self.ws + r"\}"

    def format_object_with_required_properties(
        self, property_details: List[Dict], ctx: Context
    ):
        is_required = [prop["is_required"] for prop in property_details]
        last_required_pos = max(i for i, value in enumerate(is_required) if value)
        inner = ""
        for i, pd in enumerate(property_details):
            subregex = self.format_property_kv(
                pd["key_pattern"], pd["value_pattern"], ctx
            )
            if i < last_required_pos:
                subregex = f"{subregex}{self.ws},"
            elif i > last_required_pos:
                subregex = f"{self.ws},{subregex}"
            inner += subregex if is_required[i] else f"({subregex})?"
        return r"\{" + inner + self.ws + r"\}"

    def format_object_with_additional_properties(
        self, value_pattern: str, ctx: Context, min_properties=None, max_properties=None
    ):
        inner = self._regex_repeat_elem(
            elem_pattern=f"({STRING}){self.ws}(:){self.ws}({value_pattern})",
            separator_pattern=f"{self.ws},{self.ws}",
            min_elem=min_properties,
            max_elem=max_properties,
            pad=self.ws,
        )
        return r"\{" + inner + r"\}"

    def format_array(
        self, elem_pattern: str, ctx: Context, min_items=None, max_items=None
    ):
        inner = self._regex_repeat_elem(
            elem_pattern=elem_pattern,
            separator_pattern=f"{self.ws},{self.ws}",
            min_elem=min_items,
            max_elem=max_items,
            pad=self.ws,
        )
        return rf"\[{inner}\]"

    def format_prefixItems(
        self,
        prefix_patterns: List[str],
        ctx: Context,
        suffix_elem_pattern: Optional[str] = None,
    ):
        comma_split_pattern = rf"{self.ws},{self.ws}"
        prefix_pattern = f"{self.ws}{comma_split_pattern.join(prefix_patterns)}"
        if suffix_elem_pattern:
            suffix_pattern = self._regex_repeat_elem(
                elem_pattern=suffix_elem_pattern,
                separator_pattern=f"{self.ws},{self.ws}",
                min_elem=1,
                pad=self.ws,
            )
            suffix_pattern = f"((,{suffix_pattern})|)"
            inner = f"{prefix_pattern}{suffix_pattern}"
        else:
            inner = prefix_pattern + self.ws
        return rf"\[{inner}\]"

    def format_literal(self, literal: Any):
        if isinstance(literal, str):
            return f"{re.escape(json.dumps(literal))}"
        if type(literal) in [int, bool, type(None)]:
            return re.escape(json.dumps(literal))
        elif isinstance(literal, float):
            if float(literal) == int(literal):
                int_literal = re.escape(json.dumps(int(literal)))
                float_literal = re.escape(json.dumps(float(literal)))
                return f"({int_literal}|{float_literal})"
            else:
                return re.escape(json.dumps(literal))
        else:
            raise NotImplementedError(
                f"Unsupported data type in literal: {type(literal)}. Please open an Outlines issue."
            )

    def format_allOf(self, patterns: List[str]):
        return (
            "(" + "".join([f"(?={pat})" for pat in patterns[:-1]]) + patterns[-1] + ")"
        )

    def format_anyOf(self, patterns: List[str]):
        return "(" + "|".join([f"({pat})" for pat in patterns]) + ")"

    def format_oneOf(self, patterns: List[str]):
        # If you're searching "NotImplementedError", this method also needs to be properly implemented!
        warnings.warn(
            "JSON Schema `oneOf` not implemented. Using `anyOf` instead. Please open an Outlines Issue."
        )
        return self.format_anyOf(patterns)

    def visit_unconstrained(self, node: Any, ctx: Context):
        legal_types = [
            {"type": "boolean"},
            {"type": "null"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "string"},
        ]
        allowed_nesting = node.get(
            "_allowed_nesting", ctx.nesting_level + self.max_nesting_level
        )
        # We limit the object depth to keep the expression finite, but the "depth"
        # key is not a true component of the JSON Schema specification.
        if ctx.nesting_level < allowed_nesting:
            legal_types.append({"type": "object", "_allowed_nesting": allowed_nesting})
            legal_types.append({"type": "array", "_allowed_nesting": allowed_nesting})

        subpatterns = [self.to_regex(t, ctx) for t in legal_types]
        return self.format_anyOf(subpatterns)

    def _regex_repeat_elem(
        self,
        elem_pattern: str,
        separator_pattern: str,
        min_elem=None,
        max_elem=None,
        pad="",
    ):
        """
        Creates a pattern allowing between min_elem and max_elem occurrences of elem_pattern
        Ensures each element pattern is separated by separator_pattern
        Surrounds result with `pad`
        """
        if str(max_elem) == "0":
            return pad

        base_pattern = f"({elem_pattern})"
        suffix_pattern = f"(({separator_pattern})({elem_pattern}))"

        min_suffix_repeats = "" if min_elem is None else max(0, int(min_elem) - 1)
        max_suffix_repeats = "" if max_elem is None else max_elem - 1

        if str(max_suffix_repeats) == "0":
            pattern = base_pattern
        else:
            pattern = f"{base_pattern}({suffix_pattern}){{{min_suffix_repeats},{max_suffix_repeats}}}"

        padded_pattern = f"({pad}{pattern}{pad})"

        if not min_elem:
            return f"({padded_pattern}|{pad})"
        else:
            return padded_pattern


class YAMLRegexGenerator(JSONSchemaRegexGenerator):
    """
    Core differences between JSON and YAML
    --------------------------------------

    For most types including `boolean`, `null`, `number`, and `integer`
    YAML supports a superset of JSON representation. For example, `boolean` can
    be `true` / `false` like JSON, however it can also be `yes` / `no`. For these
    types we will limit generation to the valid JSON-representation subset.

    ```
    string:
    - Equivalent to JSON, but doesn't use quotes

    array:
    - In YAML arrays are represented
    - by newline separated
    - dash-prefixed array elements

    object:
    - An object is represented as a newline separated list of key: value pairs
    ```
    """

    FORMAT_PRIMATIVE = {
        # yaml allows for more escape types
        "string": r'([^"\\\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]|\\["\\])',
        **JSONSchemaRegexGenerator.FORMAT_PRIMATIVE,
    }

    @staticmethod
    def _indentation(nesting_level: int):
        return r"(\n)" + (f"[ ]{{{nesting_level * 2}}}" if nesting_level else "")

    def format_property_kv(
        self, key_pattern: str, value_pattern: str, ctx: Context
    ) -> str:
        """
        Similar to JSON property kv, but with changes to accomodate yaml rules:
        - leading spaces are not allowed as the spaces are syntactic
        - `foo:bar` isn't a legal kv,
          - need a single space, e.g. `foo: bar`
          - or an indented newline, e.g. `foo:\n  bar`
        """
        child_indentation = self._indentation(ctx.nesting_level + 1)
        return f"({key_pattern}{self.ws}(:){child_indentation}{value_pattern})"

    def format_object_properties_all_optional(
        self, property_details: List[Dict], ctx: Context
    ):
        property_subregexes = [
            self.format_property_kv(pd["key_pattern"], pd["value_pattern"], ctx)
            for pd in property_details
        ]
        indentation = self._indentation(ctx.nesting_level)
        possible_patterns = [
            indentation.join(combination)  # first indent is optional
            for i in range(1, len(property_subregexes) + 1)
            for combination in itertools.combinations(property_subregexes, i)
        ]
        one_or_more_pattern = "|".join(possible_patterns)
        return f"({one_or_more_pattern}|{self.format_empty_object()})"

    def format_object_with_required_properties(
        self, property_details: List[Dict], ctx: Context
    ):
        is_required = [prop["is_required"] for prop in property_details]
        last_required_pos = max(i for i, value in enumerate(is_required) if value)

        indentation = self._indentation(ctx.nesting_level)

        inner = ""
        for i, pd in enumerate(property_details):
            subregex = self.format_property_kv(
                pd["key_pattern"], pd["value_pattern"], ctx
            )
            if i < last_required_pos:
                subregex = f"{subregex}{indentation}"
            elif i > last_required_pos:
                subregex = f"{indentation}{subregex}"
            inner += subregex if is_required[i] else f"({subregex})?"

        return inner

    def format_object_with_additional_properties(
        self, value_pattern: str, ctx: Context, min_properties=None, max_properties=None
    ):
        if min_properties in (0, "0", "", None):
            min_properties = 0

        inner = self._regex_repeat_elem(
            elem_pattern=self.format_property_kv(STRING, value_pattern, ctx),
            separator_pattern=self._indentation(ctx.nesting_level),
            min_elem=max(1, min_properties),
            max_elem=max_properties,
        )
        if min_properties == 0:
            empty_obj_pattern = self.format_empty_object()
            return f"({inner})|({empty_obj_pattern})"

        return inner

    def format_array(
        self, elem_pattern: str, ctx: Context, min_items=None, max_items=None
    ):
        self_indentation = self._indentation(ctx.nesting_level)

        child_indentation = self._indentation(ctx.nesting_level + 1)
        child_separator = f"([ ]|({child_indentation}))"

        if min_items in (0, "0", "", None):
            min_items = 0

        inner = self._regex_repeat_elem(
            elem_pattern=f"(-){child_separator}{elem_pattern}",
            separator_pattern=self_indentation,
            min_elem=max(1, min_items),
            max_elem=max_items,
        )
        if min_items == 0:
            empty_list_pattern = r"(\[\])"
            return f"({inner})|({empty_list_pattern})"
        return inner

    def format_prefixItems(
        self,
        prefix_patterns: List[str],
        ctx: Context,
        suffix_elem_pattern: Optional[str] = None,
    ):
        self_indentation = self._indentation(ctx.nesting_level)

        child_indentation = self._indentation(ctx.nesting_level + 1)
        child_separator = f"([ ]|({child_indentation}))"

        prefix_pattern = self_indentation.join(
            [f"(-){child_separator}{pat}" for pat in prefix_patterns]
        )

        if suffix_elem_pattern:
            suffix_pattern = self._regex_repeat_elem(
                elem_pattern=f"(-){child_separator}{suffix_elem_pattern}",
                separator_pattern=self_indentation,
                min_elem=1,
            )
            suffix_pattern = f"({self_indentation}{suffix_pattern})?"
            return f"{prefix_pattern}{suffix_pattern}"
        else:
            return prefix_pattern
