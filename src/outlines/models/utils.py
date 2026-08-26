from typing import Any


def split_multimodal_input(content: list[Any]) -> tuple[str, list[Any]]:
    """Split a multimodal input into its text prompt and assets."""
    if not content:
        raise ValueError("Content list cannot be empty.")
    if not isinstance(content[0], str):
        raise ValueError("The first item in the content list must be a string.")
    return content[0], content[1:]


def set_additional_properties_false_json_schema(schema: dict) -> dict:
    """Set additionalProperties to False on all object schemas.

    Recursively walks the JSON schema and, for every object schema, sets
    ``additionalProperties`` to False unless it is already present. An object
    schema is one whose ``type`` is ``"object"`` or, for a nullable object, a
    list containing ``"object"`` (e.g. ``["object", "null"]``).

    Parameters
    ----------
    schema
        The JSON schema to modify

    Returns
    -------
    dict
        The modified schema with additionalProperties set to False
    """

    def _walk(node):
        if isinstance(node, dict):
            # Only an object *schema* should get ``additionalProperties``. Keying
            # off the string ``"object"`` appearing anywhere would wrongly add the
            # keyword to non-object schemas, e.g. a string field with
            # ``const``/``default``/``title`` equal to ``"object"``, which the
            # OpenAI API then rejects. ``type`` may be the bare string or a list
            # such as ``["object", "null"]`` for a nullable object.
            node_type = node.get("type")
            is_object = node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            )
            if is_object and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return schema
