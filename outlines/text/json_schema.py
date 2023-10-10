import itertools
import json
import re
from typing import Callable, Dict

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
    schedule = build_schedule_from_schema(schema)

    regex = ""
    for step in schedule:
        regex += match_step_to_regex(step)

    return regex


def _ref_resolver(schema: Dict) -> Callable[[str], Dict]:
    cache: Dict[str, Dict] = dict()

    if "$id" in schema:
        cache[schema["$id"]] = schema

    if "$defs" in schema:
        for definition, annotation in schema["$defs"].items():
            cache[f"#/$defs/{definition}"] = annotation

            if "$id" in annotation:
                cache[annotation["$id"]] = annotation

    def resolver(reference: str) -> Dict:
        """Resolve a $ref reference in the context of the top-level schema."""

        if reference in cache:
            return cache[reference]

        path = reference.split("/")

        # Navigate through the top-level schema based on the path
        subschema = schema

        if path[0] != "#":
            raise ValueError(f"Unable to resolve reference: {reference}")

        for step in path[1:]:
            if step in subschema:
                subschema = subschema[step]
            else:
                raise ValueError(f"Unable to resolve reference: {reference}")

        cache[reference] = subschema
        return subschema

    return resolver


def build_schedule_from_schema(schema: str):
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
    schema = json.loads(schema)

    schema = expand_json_schema(schema, resolver=_ref_resolver(schema))
    schedule = build_schedule_from_instance(schema)

    # Concatenate adjacent strings
    reduced_schedule = [
        x
        for cls, grp in itertools.groupby(schedule, type)
        for x in (("".join(grp),) if cls is str else grp)
    ]

    return reduced_schedule


def expand_item_json_schema(expanded_property: Dict, resolver: Callable[[str], Dict]):
    """Recursively expand "$ref"s in "item"s."""
    if "items" not in expanded_property.keys():
        return
    elif "$ref" in expanded_property["items"]:
        expanded_property["items"] = expand_json_schema(
            resolver(expanded_property["items"]["$ref"]), resolver
        )
    else:
        expand_item_json_schema(expanded_property["items"], resolver)


def expand_json_schema(
    raw_schema: Dict,
    resolver: Callable[[str], Dict],
):
    """Replace references by their value in the JSON Schema.

    This recursively follows the references to other schemas in case
    of nested models. Other schemas that may exist at a higher level
    within the overall schema may be referenced via the `$ref` keyword
    according to the JSON Schema specification.


    Parameters
    ---------
    raw_schema
        The raw JSON schema as a Python dictionary, possibly with definitions
        and references.
    resolver
        A function that takes a reference and returns the corresponding schema
        or subschema from the currently scoped top-level schema.

    Returns
    -------
    A dictionary that represents the flattened equivalent of the input
    JSON schema.

    """
    expanded_properties = {}

    if "properties" in raw_schema:
        if "$id" in raw_schema:
            # see https://json-schema.org/understanding-json-schema/structuring#bundling
            resolver = _ref_resolver(raw_schema)

        for name, value in raw_schema["properties"].items():
            if "$ref" in value:  # if item is a single element
                expanded_properties[name] = expand_json_schema(
                    resolver(value["$ref"]), resolver
                )
            elif "type" in value and value["type"] == "array":  # if item is a list
                expanded_properties[name] = value

                if "$ref" in value["items"] or (
                    "type" in value["items"] and value["items"]["type"] == "array"
                ):
                    expand_item_json_schema(expanded_properties[name], resolver)
                else:
                    expanded_properties[name]["items"] = value["items"]

            else:
                expanded_properties[name] = value

        return {
            **({"title": raw_schema["title"]} if "title" in raw_schema else {}),
            "type": raw_schema["type"],
            "properties": expanded_properties,
        }

    else:
        return raw_schema


def build_schedule_from_instance(instance: Dict):
    """Build a generation schedule from a instance.

    This recursively follows the references to other instances.

    Parameters
    ----------
    instance
        An instance, can be the JSON schema itself.
    indent
        The current indentation level

    Returns
    -------
    A generation schedule for the instance, a list of strings that represent
    the structure of the JSON schema and dictionaries that contain the
    instance definition.

    """
    schedule = []
    if "properties" in instance:
        schedule.append(r"\{")
        schedule += build_schedule_from_instance(instance["properties"])
        schedule.append(r"\}")
    else:
        for i, (name, annotation) in enumerate(instance.items()):
            whitespace = r"[\n ]*"
            schedule.append(f'{whitespace}"{name}"{whitespace}:{whitespace}')

            if "anyOf" in annotation:
                schedule.append(annotation)
            elif annotation["type"] == "object":
                schedule += build_schedule_from_instance(annotation)
            else:
                schedule.append(annotation)

            # We cannot add commas after the last key-value pair in JSON
            if i == len(instance) - 1:
                schedule.append(whitespace)
            else:
                schedule.append(f"{whitespace},")

    return schedule


def match_step_to_regex(step):
    """Translate an element of a JSON schema to a regex that defines its content.

    Parameters
    ----------
    step:
        A string that represents the schema's structure, or a dictionary
        that represents a field in the schema.

    Returns
    -------
    A string that represents a regular expression that defines the value of the
    schedule's step.

    """
    if isinstance(step, str):
        return step

    if isinstance(step, dict):
        keys = set(step.keys())

        if all(key in keys for key in ("enum", "type")) and step["type"] == "string":
            choices = [f'"{re.escape(choice)}"' for choice in step["enum"]]
            return f"({'|'.join(choices)})"

        elif "enum" in keys:
            choices = [re.escape(str(choice)) for choice in step["enum"]]
            return f"({'|'.join(choices)})"

        elif all(key in keys for key in ("type", "items")) and step["type"] == "array":
            item_regexes = match_step_to_regex(step["items"])
            return rf"\[({item_regexes})(,({item_regexes}))*\]"

        elif "type" in keys and step["type"] == "object":
            steps = build_schedule_from_schema(json.dumps(step))
            regex_str = ""
            for step in steps:
                regex_str += match_step_to_regex(step)
            return regex_str

        elif (
            all(key in keys for key in ("type", "maxLength"))
            and step["type"] == "string"
        ):
            max_length = step["maxLength"]
            return f'"{STRING_INNER}{{,{max_length}}}"'

        elif (
            all(key in keys for key in ("type", "minLength"))
            and step["type"] == "string"
        ):
            min_length = step["minLength"]
            return f'"{STRING_INNER}{{{min_length},}}"'

        elif "type" in keys:
            return type_to_regex[step["type"]]

        elif "anyOf" in keys:
            regexes = [match_step_to_regex(choice) for choice in step["anyOf"]]
            return rf"({'|'.join(regexes)})"

    raise NotImplementedError
