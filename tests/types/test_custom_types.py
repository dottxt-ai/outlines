import re

import pytest
from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import to_regex


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
        (types.integer, "-19", True),
        (types.integer, "19", True),
        (types.integer, "019", False),
        (types.integer, "1.9", False),
        (types.integer, "a", False),
        (types.boolean, "True", True),
        (types.boolean, "False", True),
        (types.boolean, "true", False),
        (types.number, "10", True),
        (types.number, "10.9", True),
        (types.number, "10.9e+3", True),
        (types.number, "10.9e-3", True),
        (types.number, "a", False),
        (types.date, "2022-03-23", True),
        (types.date, "2022-03-32", False),
        (types.date, "2022-13-23", False),
        (types.date, "32-03-2022", False),
        (types.time, "01:23:59", True),
        (types.time, "01:23:61", False),
        (types.time, "01:61:59", False),
        (types.time, "24:23:59", False),
        (types.sentence, "The temperature is 23.5 degrees !", True),
        (types.sentence, "Did you earn $1,234.56 last month  ?", True),
        (types.sentence, "The #1 player scored 100 points .", True),
        (types.sentence, "Hello @world, this is a test!", True),
        (types.sentence, "invalid sentence.", False),
        (types.sentence, "Invalid sentence", False),
        (types.paragraph, "This is a paragraph!\n", True),
        (types.paragraph, "Line1\nLine2", False),
        (types.paragraph, "One sentence. Two sentences.\n\n", True),
        (types.paragraph, "One sentence. invalid sentence.", False),
        (types.paragraph, "One sentence. Invalid sentence\n", False),
        (types.hex_str, "0x123", True),
        (types.hex_str, "0xABC", True),
        (types.hex_str, "0xabc", True),
        (types.hex_str, "0x123ABC", True),
        (types.hex_str, "123", True),
        (types.hex_str, "ABC", True),
        (types.hex_str, "abc", True),
        (types.hex_str, "123ABC", True),
        (types.hex_str, "0xg123", False),
        (types.hex_str, "0x", False),
        (types.hex_str, "0x123G", False),
        (types.uuid4, "123e4567-e89b-42d3-a456-426614174000", True),
        (types.uuid4, "00000000-0000-4000-8000-000000000000", True),
        (types.uuid4, "123e4567-e89b-12d3-a456-426614174000", False),
        (types.uuid4, "123e4567-e89b-12d3-a456-42661417400", False),
        (types.uuid4, "123e4567-e89b-12d3-a456-4266141740000", False),
        (types.uuid4, "123e4567-e89b-12d3-x456-426614174000", False),
        (types.uuid4, "123e4567-e89b-12d3-a456-42661417400g", False),
        (types.ipv4, "192.168.1.1", True),
        (types.ipv4, "10.0.0.1", True),
        (types.ipv4, "172.16.0.1", True),
        (types.ipv4, "255.255.255.255", True),
        (types.ipv4, "0.0.0.0", True),
        (types.ipv4, "256.1.2.3", False),
        (types.ipv4, "1.256.2.3", False),
        (types.ipv4, "1.2.256.3", False),
        (types.ipv4, "1.2.3.256", False),
        (types.ipv4, "1.2.3", False),
        (types.ipv4, "1.2.3.4.5", False),
        (types.ipv4, "1.2.3.4.", False),
        (types.ipv4, ".1.2.3.4", False),
        (types.ipv4, "1..2.3.4", False),
    ],
)
def test_type_regex(custom_type, test_string, should_match):
    class Model(BaseModel):
        attr: custom_type

    schema = Model.model_json_schema()
    assert schema["properties"]["attr"]["type"] == "string"
    regex_str = schema["properties"]["attr"]["pattern"]
    does_match = re.fullmatch(regex_str, test_string) is not None
    assert does_match is should_match

    regex_str = to_regex(custom_type)
    does_match = re.fullmatch(regex_str, test_string) is not None
    assert does_match is should_match


@pytest.mark.parametrize(
    "custom_type,test_string,should_match",
    [
        (types.airports.IATA, "CDG", True),
        (types.airports.IATA, "XXX", False),
        (types.countries.Alpha2, "FR", True),
        (types.countries.Alpha2, "XX", False),
        (types.countries.Alpha3, "UKR", True),
        (types.countries.Alpha3, "XXX", False),
        (types.countries.Numeric, "004", True),
        (types.countries.Numeric, "900", False),
        (types.countries.Name, "Ukraine", True),
        (types.countries.Name, "Wonderland", False),
        (types.countries.Flag, "🇿🇼", True),
        (types.countries.Flag, "🤗", False),
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

    does_match = test_string in custom_type.__members__
    assert does_match is should_match
