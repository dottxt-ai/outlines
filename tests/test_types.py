import re

import pytest
from pydantic import BaseModel

from outlines import outline_types
from outlines.fsm.types import python_types_to_regex


@pytest.mark.parametrize(
    "custom_type,test_string,should_match",
    [
        (outline_types.phone_numbers.USPhoneNumber, "12", False),
        (outline_types.phone_numbers.USPhoneNumber, "(123) 123-1234", True),
        (outline_types.phone_numbers.USPhoneNumber, "123-123-1234", True),
        (outline_types.zip_codes.USZipCode, "12", False),
        (outline_types.zip_codes.USZipCode, "12345", True),
        (outline_types.zip_codes.USZipCode, "12345-1234", True),
        (outline_types.ISBN, "ISBN 0-1-2-3-4-5", False),
        (outline_types.ISBN, "ISBN 978-0-596-52068-7", True),
        # (types.ISBN, "ISBN 978-0-596-52068-1", True), wrong check digit
        (outline_types.ISBN, "ISBN-13: 978-0-596-52068-7", True),
        (outline_types.ISBN, "978 0 596 52068 7", True),
        (outline_types.ISBN, "9780596520687", True),
        (outline_types.ISBN, "ISBN-10: 0-596-52068-9", True),
        (outline_types.ISBN, "0-596-52068-9", True),
        (outline_types.Email, "eitan@gmail.com", True),
        (outline_types.Email, "99@yahoo.com", True),
        (outline_types.Email, "eitan@.gmail.com", False),
        (outline_types.Email, "myemail", False),
        (outline_types.Email, "eitan@gmail", False),
        (outline_types.Email, "eitan@my.custom.domain", True),
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


def test_locale_not_implemented():
    with pytest.raises(NotImplementedError):
        outline_types.locale("fr")


@pytest.mark.parametrize(
    "locale_str,base_types,locale_types",
    [
        (
            "us",
            ["ZipCode", "PhoneNumber"],
            [outline_types.zip_codes.USZipCode, outline_types.phone_numbers.USPhoneNumber],
        )
    ],
)
def test_locale(locale_str, base_types, locale_types):
    for base_type, locale_type in zip(base_types, locale_types):
        type = getattr(outline_types.locale(locale_str), base_type)
        assert type == locale_type


@pytest.mark.parametrize(
    "custom_type,test_string,should_match",
    [
        (outline_types.airports.IATA, "CDG", True),
        (outline_types.airports.IATA, "XXX", False),
        (outline_types.countries.Alpha2, "FR", True),
        (outline_types.countries.Alpha2, "XX", False),
        (outline_types.countries.Alpha3, "UKR", True),
        (outline_types.countries.Alpha3, "XXX", False),
        (outline_types.countries.Numeric, "004", True),
        (outline_types.countries.Numeric, "900", False),
        (outline_types.countries.Name, "Ukraine", True),
        (outline_types.countries.Name, "Wonderland", False),
        (outline_types.countries.Flag, "🇿🇼", True),
        (outline_types.countries.Flag, "🤗", False),
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
