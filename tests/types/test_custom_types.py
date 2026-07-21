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
        (types.number, "1e10", True),
        (types.number, "1E5", True),
        (types.number, "6.022e23", True),
        (types.number, "1.5e8", True),
        (types.number, "1e", False),
        (types.number, "a", False),
        (types.date, "2022-03-23", True),
        (types.date, "2022-03-01", True),
        (types.date, "2022-03-31", True),
        (types.date, "2022-03-32", False),
        (types.date, "2022-03-00", False),
        (types.date, "2022-13-23", False),
        (types.date, "2022-00-23", False),
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
        (types.ipv4, "01.1.1.1", False),
        (types.ipv4, "00.0.0.0", False),
        (types.ipv4, "1.02.3.4", False),
        (types.ipv4, "1.1.1.09", False),
        (types.ipv4, "1.2.3.4.5", False),
        (types.ipv4, "1.2.3.4.", False),
        (types.ipv4, ".1.2.3.4", False),
        (types.ipv4, "1..2.3.4", False),
        (types.ipv6, "2001:0db8:85a3:0000:0000:8a2e:0370:7334", True),
        (types.ipv6, "2001:db8::1", True),
        (types.ipv6, "::1", True),
        (types.ipv6, "::", True),
        (types.ipv6, "fe80::1", True),
        (types.ipv6, "::ffff:192.168.1.1", True),
        (types.ipv6, "2001:db8::192.168.1.1", True),
        (types.ipv6, "2001:db8:85a3:0:0:8a2e:370:7334:1", False),
        (types.ipv6, "2001::db8::1", False),
        (types.ipv6, "12345::1", False),
        (types.ipv6, ":::1", False),
        (types.ipv6, "::ffff:01.1.1.1", False),
        (types.ipv6, "::ffff:00.0.0.0", False),
        (types.ipv6, "1:2:3:4::5.6.7.8", True),
        (types.semver, "1.2.3", True),
        (types.semver, "1.2", False),
        (types.semver, "01.2.3", False),
        (types.semver, "1.2.3-alpha+001", True),
        (types.mac_address, "00:1A:2B:3C:4D:5E", True),
        (types.mac_address, "00-1A-2B-3C-4D-5E", False),
        (types.mac_address, "00:1A:2B:3C:4D", False),
        (types.hex_color, "#1a2b3c", True),
        (types.hex_color, "#abc", True),
        (types.hex_color, "#ABC123", True),
        (types.hex_color, "1a2b3c", False),
        (types.hex_color, "#12345", False),
        (types.hex_color, "#gggggg", False),
        (types.hex_color, "#abcd", False),
        (types.hex_color, "#aabbccdd", False),
        (types.hex_color, " #abc", False),
        (types.slug, "my-post-title", True),
        (types.slug, "post", True),
        (types.slug, "a1-b2", True),
        (types.slug, "My-Post", False),
        (types.slug, "-leading-hyphen", False),
        (types.slug, "trailing-hyphen-", False),
        (types.slug, "double--hyphen", False),
        (types.slug, "under_score", False),
        (types.slug, "", False),
        (types.credit_card, "4111111111111111", True),  # Visa 16
        (types.credit_card, "4222222222222", True),  # Visa 13
        (types.credit_card, "4111111111111111111", True),  # Visa 19
        (types.credit_card, "5555555555554444", True),  # Mastercard 51-55
        (types.credit_card, "2223003122003222", True),  # Mastercard 2221-2720
        (types.credit_card, "378282246310005", True),  # American Express
        (types.credit_card, "30569309025904", True),  # Diners Club
        (types.credit_card, "6011111111111117", True),  # Discover 6011
        (types.credit_card, "6440000000000007", True),  # Discover 644-649
        (types.credit_card, "6490000000000009", True),  # Discover 644-649
        (types.credit_card, "6512345678901234", True),  # Discover 65
        (types.credit_card, "3530111333300000", True),  # JCB
        (types.credit_card, "6759649826438453", True),  # Maestro
        (types.credit_card, "6200000000000005", True),  # UnionPay
        (types.credit_card, "6221260000000000", True),  # Discover 622126-622925 co-brand
        (types.credit_card, "4111 1111 1111 1111", False),  # spaces
        (types.credit_card, "4111-1111-1111-1111", False),  # hyphens
        (types.credit_card, "1234567890123456", False),  # unknown prefix
        (types.credit_card, "6430000000000000", False),  # 643 not a Discover prefix
        (types.credit_card, "2220003122003222", False),  # below Mastercard 2-range
        (types.credit_card, "2721003122003222", False),  # above Mastercard 2-range
        (types.credit_card, "411111111111", False),  # too short
        (types.credit_card, "4111a11111111111", False),  # non-digit character
        (types.credit_card, "41111111111111111111", False),  # too long (20 digits)
        (types.credit_card, "3782822463100050", False),  # Amex prefix, wrong length
        (types.credit_card, "", False),
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
