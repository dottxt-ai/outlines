import re

import pytest
from pydantic import BaseModel

from outlines import types
from outlines.fsm.types import python_types_to_regex


@pytest.mark.parametrize(
    "custom_type,test_string,should_match",
    [
        (types.locale.us.phone_number, "12", False),
        (types.locale.us.phone_number, "(123) 123-1234", True),
        (types.locale.us.phone_number, "123-123-1234", True),
        (types.locale.us.zip_code, "12", False),
        (types.locale.us.zip_code, "12345", True),
        (types.locale.us.zip_code, "12345-1234", True),
        (types.isbn, "ISBN 0-1-2-3-4-5", False),
        (types.isbn, "ISBN 978-0-596-52068-7", True),
        # (types.ISBN, "ISBN 978-0-596-52068-1", True), wrong check digit
        (types.isbn, "ISBN-13: 978-0-596-52068-7", True),
        (types.isbn, "978 0 596 52068 7", True),
        (types.isbn, "9780596520687", True),
        (types.isbn, "ISBN-10: 0-596-52068-9", True),
        (types.isbn, "0-596-52068-9", True),
        (types.email, "eitan@gmail.com", True),
        (types.email, "99@yahoo.com", True),
        (types.email, "eitan@.gmail.com", False),
        (types.email, "myemail", False),
        (types.email, "eitan@gmail", False),
        (types.email, "eitan@my.custom.domain", True),
    ],
)
def test_type_regex(custom_type, test_string, should_match):
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


@pytest.mark.parametrize(
    "custom_type,test_string,should_match",
    [
        (types.airports.IATA, "CDG", True),
        (types.airports.IATA, "XXX", False),
        (types.countries.alpha2, "FR", True),
        (types.countries.alpha2, "XX", False),
        (types.countries.alpha3, "UKR", True),
        (types.countries.alpha3, "XXX", False),
        (types.countries.numeric, "004", True),
        (types.countries.numeric, "900", False),
        (types.countries.name, "Ukraine", True),
        (types.countries.name, "Wonderland", False),
        (types.countries.flag, "🇿🇼", True),
        (types.countries.flag, "🤗", False),
    ],
)
def test_type_enum(custom_type, test_string, should_match):
    type_name = custom_type.__name__

    class Model(BaseModel):
        attr: custom_type

    schema = Model.model_json_schema()
    assert isinstance(schema["$defs"][type_name]["enum"], list)
    does_match = test_string in schema["$defs"][type_name]["enum"]
    assert does_match is should_match

    regex_str, format_fn = python_types_to_regex(custom_type)
    assert isinstance(format_fn(1), str)
    does_match = re.match(regex_str, test_string) is not None
    assert does_match is should_match
