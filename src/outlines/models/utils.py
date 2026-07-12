import jsonpath_ng


def set_additional_properties_false_json_schema(schema: dict) -> dict:
    """Set additionalProperties to False to all objects in the schema using jsonpath.

    Parameters
    ----------
    schema
        The JSON schema to modify

    Returns
    -------
    dict
        The modified schema with additionalProperties set to False
    """
    # Get all nodes
    jsonpath_expr = jsonpath_ng.parse('$..*')
    matches = jsonpath_expr.find(schema)

    # Go over all nodes and set additionalProperties to False if it's an
    # object. `type` can either be the bare string "object" or, per the JSON
    # Schema spec, a list of type names (e.g. ["object", "null"]) used to
    # express nullable objects, so both forms must be checked.
    for match in matches:
        is_object_type = match.value == 'object' or (
            isinstance(match.value, list) and 'object' in match.value
        )
        if is_object_type:
            if 'additionalProperties' not in match.context.value:
                match.context.value['additionalProperties'] = False

    return schema
