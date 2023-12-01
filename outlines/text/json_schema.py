import inspect
import itertools as it
import json
import re
from typing import Callable, Union

from jsonschema.protocols import Validator
from pydantic import BaseModel, create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012

STRING_INNER = r'(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)'
STRING = f'"{STRING_INNER}*"'
INTEGER = r"(0|[1-9][0-9]*)"
NUMBER = rf"(-)?({INTEGER})(\.[0-9]+)?([eE][+-][0-9]+)?"
BOOLEAN = r"(true|false)"
NULL = r"null"

type_to_regex = {
    "string": STRING,
    "integer": INTEGER,
    "number": NUMBER,
    "boolean": BOOLEAN,
    "null": NULL,
}


def build_regex_from_object(object: Union[str, Callable, BaseModel]):
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

    Returns
    -------
    A generation schedule. A list of strings that represent the JSON
    schema's structure and regular expression that define the structure of
    the fields.

    References
    ----------
    .. [0] JSON Schema. https://json-schema.org/

    """

    if isinstance(object, type(BaseModel)):
        schema = object.model_json_schema()
    elif callable(object):
        schema = get_schema_from_signature(object)
    else:
        schema = json.loads(object)

    Validator.check_schema(schema)

    # Build reference resolver
    schema = Resource(contents=schema, specification=DRAFT202012)
    uri = schema.id() if schema.id() is not None else ""
    registry = Registry().with_resource(uri=uri, resource=schema)
    resolver = registry.resolver()

    content = schema.contents
    return to_regex(resolver, content)


def to_regex(resolver: Resolver, instance: dict):
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
    whitespace = r"[\n ]*"

    if "properties" in instance:
        regex = ""
        regex += r"\{"
        for i, (name, value) in enumerate(instance["properties"].items()):
            regex += f'{whitespace}"{name}"{whitespace}:{whitespace}'
            regex += to_regex(resolver, value)

            # No comma after the last key-value pair in JSON
            if i < len(instance["properties"]) - 1:
                regex += f"{whitespace},"

        regex += f"{whitespace}" + r"\}"

        return regex

    # To validate against allOf, the given data must be valid against all of the
    # given subschemas.
    elif "allOf" in instance:
        subregexes = [to_regex(resolver, t) for t in instance["allOf"]]
        subregexes_str = [f"{subregex}" for subregex in subregexes]
        return rf"({''.join(subregexes_str)})"

    # To validate against `anyOf`, the given data must be valid against
    # any (one or more) of the given subschemas.
    elif "anyOf" in instance:
        subregexes = [to_regex(resolver, t) for t in instance["anyOf"]]
        combinations = [
            "(" + "".join(c) + ")"
            for r in range(1, len(subregexes) + 1)
            for c in it.permutations(subregexes, r)
        ]

        return rf"({'|'.join(combinations)})"

    # To validate against oneOf, the given data must be valid against exactly
    # one of the given subschemas.
    elif "oneOf" in instance:
        subregexes = [to_regex(resolver, t) for t in instance["oneOf"]]

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
        instance = resolver.lookup(path).contents
        return to_regex(resolver, instance)

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
            min_items = instance.get("minItems", "0")
            max_items = instance.get("maxItems", "")
            if min_items == max_items:
                num_repeats = "{" + str(int(min_items) - 1) + "}"
            else:
                num_repeats = "*"

            if "items" in instance:
                items_regex = to_regex(resolver, instance["items"])
                return rf"\[({items_regex})(,({items_regex})){num_repeats}\]"
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
                regexes = [to_regex(resolver, t) for t in types]
                return (
                    rf"\[({'|'.join(regexes)})(,({'|'.join(regexes)})){num_repeats}\]"
                )

        elif instance_type == "boolean":
            return type_to_regex["boolean"]

        elif instance_type == "null":
            return type_to_regex["null"]

        elif isinstance(instance_type, list):
            # Here we need to make the choice to exclude generating an object
            # if the specification of the object is not give, even though a JSON
            # object that contains an object here would be valid under the specification.
            regexes = [
                to_regex(resolver, {"type": t}) for t in instance_type if t != "object"
            ]
            return rf"({'|'.join(regexes)})"

    raise NotImplementedError(
        f"""Could not translate the instance {instance} to a
    regular expression. Make sure it is valid to the JSON Schema specification. If
    it is, please open an issue on the Outlines repository"""
    )


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

    model = create_model("Arguments", **arguments)

    return model.model_json_schema()
