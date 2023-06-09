import itertools
import json
from typing import Dict

STRING = r'".*"'
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

    # Find object definitions in the schema, if any
    definitions = {}
    if "$defs" in schema:
        for definition, annotation in schema["$defs"].items():
            definitions[f"#/$defs/{definition}"] = annotation

    schema = expand_json_schema(schema, definitions)
    schedule = build_schedule_from_instance(schema)

    # Concatenate adjacent strings
    reduced_schedule = [
        x
        for cls, grp in itertools.groupby(schedule, type)
        for x in (("".join(grp),) if cls is str else grp)
    ]

    return reduced_schedule


def expand_json_schema(raw_schema: Dict, definitions: Dict):
    """Replace references by their value in the JSON Schema.

    This recursively follows the references to other schemas in case
    of nested models. Other schemas are stored under the "definitions"
    key in the schema of the top-level model.

    Parameters
    ---------
    raw_schema
        The raw JSON schema as a Python dictionary, possibly with definitions
        and references.
    definitions
        The currently known definitions.

    Returns
    -------
    A dictionary that represents the flattened equivalent of the input
    JSON schema.

    """
    expanded_properties = {}

    if "properties" in raw_schema:
        for name, value in raw_schema["properties"].items():
            if "$ref" in value:  # if item is a single element
                expanded_properties[name] = expand_json_schema(
                    definitions[value["$ref"]], definitions
                )
            elif "type" in value and value["type"] == "array":  # if item is a list
                expanded_properties[name] = value
                if "$ref" in value["items"]:
                    expanded_properties[name]["items"] = expand_json_schema(
                        definitions[value["items"]["$ref"]], definitions
                    )
                else:
                    expanded_properties[name]["items"] = value["items"]
            else:
                expanded_properties[name] = value

        return {
            "title": raw_schema["title"],
            "type": raw_schema["type"],
            "properties": expanded_properties,
        }

    else:
        return raw_schema


def build_schedule_from_instance(instance: Dict, indent: int = 0):
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
        schedule.append("{\n")
        schedule += build_schedule_from_instance(instance["properties"], indent + 2)
        if indent > 0:
            schedule.append(" " * indent)
        schedule.append("}")
    else:
        for i, (name, annotation) in enumerate(instance.items()):
            schedule.append(" " * indent)
            schedule.append(f'"{name}": ')
            if "anyOf" in annotation:
                schedule.append(annotation)
            elif annotation["type"] == "object":
                schedule += build_schedule_from_instance(annotation, indent)
            else:
                schedule.append(annotation)

            # We cannot add commas after the last key-value pair in JSON
            if i == len(instance) - 1:
                schedule.append("\n")
            else:
                schedule.append(",\n")

    return schedule


def match_step_to_regex(step):
    """Translate an element of a JSON schema to a regex that defines its content.

    Parameters
    ----------
    step:
        A string that represents the schema's structure, or a dictionnary
        that represents a field in the schema.

    Returns
    -------
    A string that represents a regular expression that defines the value of the
    schedule's step.

    """
    match step:
        case str() as step:
            return step

        case {"enum": choices, "type": "string"}:
            choices = [f'"{choice}"' for choice in choices]
            return f"({'|'.join(choices)})"
        case {"enum": choices}:
            choices = [str(choice) for choice in choices]
            return f"({'|'.join(choices)})"

        case {"type": "array", "items": items}:
            item_regexes = match_step_to_regex(items)
            return rf"\[({item_regexes})(,({item_regexes}))*\]"

        case {"type": "object"} as object:
            steps = build_schedule_from_schema(json.dumps(object))
            regex_str = ""
            for step in steps:
                regex_str += match_step_to_regex(step)
            return regex_str

        case {"type": "string", "maxLength": max_length}:
            return f'".{{,{max_length}}}"'
        case {"type": "string", "minLength": min_length}:
            return f'".{{{min_length},}}"'

        case {"type": field_type}:
            return type_to_regex[field_type]

        case {"anyOf": choices}:
            regexes = [match_step_to_regex(choice) for choice in choices]
            return rf"({'|'.join(regexes)})"

        case _:
            raise NotImplementedError
