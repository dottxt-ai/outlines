import pytest

from outlines.types.json_schema_utils import (
    json_schema_dict_to_dataclass,
    json_schema_dict_to_pydantic,
    json_schema_dict_to_typeddict,
)

SCHEMA = {
    "type": "object",
    "properties": {"a": {"enum": []}},
    "required": ["a"],
}


@pytest.mark.parametrize(
    "convert",
    [
        json_schema_dict_to_pydantic,
        json_schema_dict_to_typeddict,
        json_schema_dict_to_dataclass,
    ],
)
def test_an_empty_enum_is_named(convert):
    with pytest.raises(ValueError, match="must contain at least one value"):
        convert(SCHEMA, "EmptyEnum")


def test_a_filled_enum_still_becomes_a_literal():
    schema = {
        "type": "object",
        "properties": {"a": {"enum": ["x", "y"]}},
        "required": ["a"],
    }

    model = json_schema_dict_to_pydantic(schema, "FilledEnum")

    assert model(a="x").a == "x"
