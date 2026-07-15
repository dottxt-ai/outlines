from typing import Any


def set_additional_properties_false_json_schema(schema: dict) -> dict:
    """Set additionalProperties to False on every object subschema.

    Parameters
    ----------
    schema
        The JSON schema to modify

    Returns
    -------
    dict
        The modified schema with additionalProperties set to False
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return schema
