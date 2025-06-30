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

    # Go over all nodes and set additionalProperties to False if it's an object
    for match in matches:
        if match.value == 'object':
            if 'additionalProperties' not in match.context.value:
                match.context.value['additionalProperties'] = False

    return schema
