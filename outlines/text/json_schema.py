import json
import re

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


def build_regex_from_schema(schema: str):
    """Turn a JSON schema into a regex that matches any JSON object that follows
    this schema.

    Parameters
    ----------
    schema
        A string that contains the JSON schema.

    Returns
    -------
    A string that contains a regular expression that matches any JSON object that
    follows the schema.

    """
    schema = json.loads(schema)

    # Build reference resolver
    schema = Resource(contents=schema, specification=DRAFT202012)
    uri = schema.id() if schema.id() is not None else ""
    registry = Registry().with_resource(uri=uri, resource=schema)
    resolver = registry.resolver()

    content = schema.contents
    regex = to_regex(resolver, content)
    return regex


def to_regex(resolver: Resolver, instance: dict):
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

    elif "oneOf" in instance:
        print(instance)

    elif "allOf" in instance:
        print(instance)

    elif "anyOf" in instance:
        subregexes = [to_regex(resolver, t) for t in instance["anyOf"]]
        return rf"({'|'.join(subregexes)})"

    elif "enum" in instance:
        if instance["type"] == "string":
            choices = [f'"{re.escape(choice)}"' for choice in instance["enum"]]
            return f"({'|'.join(choices)})"
        else:
            choices = [re.escape(str(choice)) for choice in instance["enum"]]
            return f"({'|'.join(choices)})"

    elif "$ref" in instance:
        path = f"{instance['$ref']}"
        instance = resolver.lookup(path).contents
        return to_regex(resolver, instance)

    elif "type" in instance:
        type = instance["type"]

        if type == "string":
            if "maxLength" in instance or "minLength" in instance:
                max_length = instance.get("maxLength", "")
                min_length = instance.get("minLength", "")
                try:
                    if int(max_length) < int(min_length):
                        raise ValueError(
                            "maxLength must be greater than or equal to minLength"
                        )
                except ValueError:
                    pass
                return f'"{STRING_INNER}{{{min_length},{max_length}}}"'
            elif "pattern" in instance:
                pattern = instance["pattern"]
                if pattern[0] == "^" and pattern[-1] == "$":
                    return rf'(^"{pattern[1:-1]}"$)'
                else:
                    return rf'("{pattern}")'
            else:
                return type_to_regex["string"]

        elif type == "number":
            return type_to_regex["number"]

        elif type == "integer":
            return type_to_regex["integer"]

        elif type == "array":
            items_regex = to_regex(resolver, instance["items"])
            return rf"\[({items_regex})(,({items_regex}))*\]"

        elif type == "boolean":
            return type_to_regex["boolean"]

        elif type == "null":
            return type_to_regex["null"]

        # elif isinstance(type, list):
        #     if "object" in type:
        #         expanded = to_regex(resolver, instance)
        #         return ""
        #     return ""

    raise NotImplementedError(
        f"""Could not translate the instance {instance} to a
    regular expression. Make sure it is valid to the JSON Schema specification. If
    it is, please open an issue on the Outlines repository"""
    )
