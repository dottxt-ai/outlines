import re

import pytest
from pydantic import BaseModel

from outlines import types
from outlines.fsm.types import python_types_to_regex


@pytest.mark.parametrize(
    "custom_type,test_string,should_match",
    [
        (types.PhoneNumber, "12", False),
        (types.PhoneNumber, "(123) 123-1234", True),
        (types.PhoneNumber, "123-123-1234", True),
        (types.ZipCode, "12", False),
        (types.ZipCode, "12345", True),
        (types.ZipCode, "12345-1234", True),
    ],
)
def test_phone_number(custom_type, test_string, should_match):
    class Model(BaseModel):
        attr: custom_type

    schema = Model.model_json_schema()
    assert schema["properties"]["attr"]["type"] == "string"
    regex_str = schema["properties"]["attr"]["pattern"]
    does_match = re.match(regex_str, test_string) is not None
    assert does_match is should_match

    regex_str, format_fn = python_types_to_regex(custom_type)
    assert isinstance(format_fn(1), str)
    does_match = re.match(regex_str, test_string) is not None
    assert does_match is should_match
