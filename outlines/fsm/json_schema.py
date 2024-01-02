import inspect
import itertools as it
import json
import re
from typing import Any, Callable, Type, Union

from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from pydantic import BaseModel, create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012, Schema

DEFINITIONS = {
    "__whitespace__": r"(?:[ \t\n\r]*)",
    "__json_object__": r"\{\s*(?&__members__)?\s*\}",
    "__members__": r"(?&__member__)(\s*,\s*(?&__member__))*",
    "__member__": r"(?&__string__)\s*:\s*(?&__json_value__)",
    "__json_array__": r"\[\s*((?&__json_value__)(\s*,\s*(?&__json_value__))*)?\s*\]",
    "__string_inner__": r"""(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)""",
    "__string__": r'"(?&__string_inner__)*"',
    "__number__": r"(?&__integer__)(?&__fraction__)?(?&__exponent__)?",
    "__integer__": r"-?(?:0|[1-9][0-9]*)",
    "__fraction__": r"\.[0-9]*",
    "__exponent__": r"[eE][-+]?[0-9]+",
    "__boolean__": r"true|false",
    "__null__": r"null",
    "__json_value__": r"(?&__string__)|(?&__number__)|(?&__json_object__)|(?&__json_array__)|(?&__boolean__)|(?&__null__)",
}

WHITESPACE = r"(?&__whitespace__)"
STRING_INNER = r"(?&__string_inner__)"
STRING = r"(?&__string__)"
INTEGER = r"(?&__integer__)"
NUMBER = r"(?&__number__)"
BOOLEAN = r"(?&__boolean__)"
NULL = r"(?&__null__)"
JSON_VALUE = r"(?&__json_value__)"

type_to_regex = {
    "string": STRING,
    "integer": INTEGER,
    "number": NUMBER,
    "boolean": BOOLEAN,
    "null": NULL,
}


def build_regex_from_object(
    object: Union[str, Callable, BaseModel, Type[BaseModel]]
) -> str:
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
        A JSON schema or a Python object that can be converted to a JSON schema.
        See [0] for more information.

    Returns
    -------
    A generation schedule. A list of strings that represent the JSON
    schema's structure and regular expression that define the structure of
    the fields.

    References
    ----------
    .. [0] JSON Schema. https://json-schema.org/

    """

    # Extract the schema from the object
    schema: Schema
    if isinstance(object, BaseModel):
        schema = object.model_json_schema()
    elif isinstance(object, type) and issubclass(object, BaseModel):
        schema = object.model_json_schema()
    elif callable(object):
        schema = get_schema_from_signature(object)
    else:
        schema = json.loads(object)

    # Validate the schema against the JSON Schema specification
    validator: Validator = validator_for(schema)
    validator.check_schema(schema)  # type: ignore

    # Build reference resolver
    resource = Resource(contents=schema, specification=DRAFT202012)
    uri = resource.id() or ""
    registry = Registry().with_resource(uri=uri, resource=resource)
    resolver = registry.resolver()

    content = resource.contents
    return to_regex(resolver, content)


def to_regex(resolver: None | Resolver, instance: Schema) -> str:
    """Translate a JSON Schema instance into a regex that validates the schema.

    Note
    ----
    Many features of JSON schema are missing:
    - Support the fact that fields in an object are optional by default
    - Handle `required` keyword
    - Handle `additionalProperties` keyword
    - Handle types defined as a list
    - Handle constraints on numbers
    - Handle special patterns: `date`, `uri`, etc.
    - Handle optional fields (not in `required`)

    This does not support recursive definitions.

    Parameters
    ----------
    resolver
        An object that resolves references to other instances within a schema
    instance
        The instance to translate
    """

    class Path(str):
        pass

    class Regex(str):
        pass

    definitions: dict[str, Path | Regex] = {
        name: Regex(regex) for name, regex in DEFINITIONS.items()
    }

    def go(instance: Schema) -> str:
        if isinstance(instance, bool):
            if instance:
                # True means any JSON object is valid
                return JSON_VALUE
            else:
                # False means no JSON object is valid
                return r""

        if instance == {}:
            # Empty object means any JSON object is valid
            return JSON_VALUE

        if "properties" in instance:
            regex = ""
            regex += r"\{"
            for i, (name, value) in enumerate(instance["properties"].items()):
                regex += f'{WHITESPACE}"{name}"{WHITESPACE}:{WHITESPACE}'
                regex += go(value)

                # No comma after the last key-value pair in JSON
                if i < len(instance["properties"]) - 1:
                    regex += f"{WHITESPACE},"

            regex += f"{WHITESPACE}" + r"\}"

            return regex

        # To validate against allOf, the given data must be valid against all of the
        # given subschemas.
        elif "allOf" in instance:
            subregexes = [go(t) for t in instance["allOf"]]
            subregexes_str = [f"{subregex}" for subregex in subregexes]
            return rf"({''.join(subregexes_str)})"

        # To validate against `anyOf`, the given data must be valid against
        # any (one or more) of the given subschemas.
        elif "anyOf" in instance:
            subregexes = [go(t) for t in instance["anyOf"]]
            combinations = [
                "(" + "".join(c) + ")"
                for r in range(1, len(subregexes) + 1)
                for c in it.permutations(subregexes, r)
            ]

            return rf"({'|'.join(combinations)})"

        # To validate against oneOf, the given data must be valid against exactly
        # one of the given subschemas.
        elif "oneOf" in instance:
            subregexes = [go(t) for t in instance["oneOf"]]

            xor_patterns = []
            # json schema validation ensured there is no overlapping schemas in oneOf
            for subregex in subregexes:
                other_subregexes = filter(lambda r: r != subregex, subregexes)
                other_subregexes_str = "|".join([f"{s}" for s in other_subregexes])
                negative_lookahead = f"(?!.*({other_subregexes_str}))"
                xor_patterns.append(f"({subregex}){negative_lookahead}")

            return rf"({'|'.join(xor_patterns)})"

        # The enum keyword is used to restrict a value to a fixed set of values. It
        # must be an array with at least one element, where each element is unique.
        elif "enum" in instance:
            choices = []
            for choice in instance["enum"]:
                if type(choice) in [int, float, bool, None]:
                    choices.append(re.escape(str(choice)))
                elif type(choice) == str:
                    choices.append(f'"{re.escape(choice)}"')

            return f"({'|'.join(choices)})"

        elif "$ref" in instance:
            path = f"{instance['$ref']}"
            name = re.escape(path.replace("/", "_").replace("#", "").replace("$", "_"))
            assert resolver is not None, "Cannot resolve references without a resolver"
            if name not in definitions:
                definitions[name] = Path(path)
            return f"(?&{name})"

        # The type keyword may either be a string or an array:
        # - If it's a string, it is the name of one of the basic types.
        # - If it is an array, it must be an array of strings, where each string is
        # the name of one of the basic types, and each element is unique. In this
        # case, the JSON snippet is valid if it matches any of the given types.
        elif "type" in instance:
            instance_type = instance["type"]
            if instance_type == "string":
                if "maxLength" in instance or "minLength" in instance:
                    max_items = instance.get("maxLength", "")
                    min_items = instance.get("minLength", "")
                    try:
                        if int(max_items) < int(min_items):
                            raise ValueError(
                                "maxLength must be greater than or equal to minLength"
                            )
                    except ValueError:
                        pass
                    return f'"{STRING_INNER}{{{min_items},{max_items}}}"'
                elif "pattern" in instance:
                    pattern = instance["pattern"]
                    if pattern[0] == "^" and pattern[-1] == "$":
                        return rf'(^"{pattern[1:-1]}"$)'
                    else:
                        return rf'("{pattern}")'
                else:
                    return type_to_regex["string"]

            elif instance_type == "number":
                return type_to_regex["number"]

            elif instance_type == "integer":
                return type_to_regex["integer"]

            elif instance_type == "array":
                if "items" in instance:
                    items_regex = go(instance["items"])
                else:
                    # Here we need to make the choice to exclude generating list of objects
                    # if the specification of the object is not given, even though a JSON
                    # object that contains an object here would be valid under the specification.
                    types = [
                        {"type": "boolean"},
                        {"type": "null"},
                        {"type": "number"},
                        {"type": "integer"},
                        {"type": "string"},
                    ]
                    items_regex = rf"({'|'.join(go(t) for t in types)})"

                min_items = instance.get("minItems")
                min_items = int(min_items) if min_items is not None else 0
                max_items = instance.get("maxItems")
                max_items = int(max_items) if max_items is not None else None

                if min_items == 0 and max_items is None:
                    middle = rf"({items_regex}(,{items_regex})*)?"

                elif min_items > 0 and max_items is None:
                    middle = (
                        rf"{items_regex}(,{items_regex})"
                        + r"{"
                        + rf"{min_items-1},"
                        + r"}"
                    )

                elif min_items == 0 and max_items is not None:
                    if max_items == 0:
                        middle = r""
                    else:
                        middle = (
                            rf"({items_regex}(,{items_regex})"
                            + r"{"
                            + rf"0,{max_items-1}"
                            + r"})?"
                        )

                elif min_items > 0 and max_items is not None:
                    if max_items >= min_items:
                        middle = (
                            rf"{items_regex}(,{items_regex})"
                            + r"{"
                            + rf"{min_items-1},{max_items-1}"
                            + r"}"
                        )
                    else:
                        raise ValueError(
                            "max_items must be greater than or equal to min_items"
                        )

                else:
                    raise ValueError("min_items must be greater than or equal to 0")

                return rf"\[{middle}\]"

            elif instance_type == "boolean":
                return type_to_regex["boolean"]

            elif instance_type == "null":
                return type_to_regex["null"]

            elif isinstance(instance_type, list):
                # Here we need to make the choice to exclude generating an object
                # if the specification of the object is not give, even though a JSON
                # object that contains an object here would be valid under the specification.
                regexes = [go({"type": t}) for t in instance_type if t != "object"]
                return rf"({'|'.join(regexes)})"

        raise NotImplementedError(
            f"""Could not translate the instance {instance} to a
        regular expression. Make sure it is valid to the JSON Schema specification. If
        it is, please open an issue on the Outlines repository"""
        )

    definitions["__self__"] = Regex(go(instance))

    while any(isinstance(v, Path) for v in definitions.values()):
        for name, value in definitions.items():
            if isinstance(value, Path):
                assert (
                    resolver is not None
                ), "Cannot resolve references without a resolver"
                definitions[name] = Regex(go(resolver.lookup(value).contents))

    regex = r"(?:"
    for name, value in definitions.items():
        assert isinstance(value, Regex)
        regex += rf"(?P<{name}>{value})"
    regex += r"){0}"
    regex += r"(?&__self__)"

    return regex


def get_schema_from_signature(fn: Callable) -> dict[str, Any]:
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

    model = create_model("Arguments", **arguments)

    return model.model_json_schema()
