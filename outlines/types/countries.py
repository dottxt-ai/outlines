"""Generate valid country codes and names."""

from enum import Enum

from iso3166 import countries


def get_country_flags():
    """Generate Unicode flags for all ISO 3166-1 alpha-2 country codes in Alpha2 Enum."""
    base = ord("🇦")
    return {
        code.name: chr(base + ord(code.name[0]) - ord("A"))
        + chr(base + ord(code.name[1]) - ord("A"))
        for code in Alpha2
    }


ALPHA_2_CODE = [(country.alpha2, country.alpha2) for country in countries]
Alpha2 = Enum("Alpha_2", ALPHA_2_CODE)  # type:ignore

ALPHA_3_CODE = [(country.alpha3, country.alpha3) for country in countries]
Alpha3 = Enum("Alpha_3", ALPHA_3_CODE)  # type:ignore

NUMERIC_CODE = [(str(country.numeric), str(country.numeric)) for country in countries]
Numeric = Enum("Numeric_code", NUMERIC_CODE)  # type:ignore

NAME = [(country.name, country.name) for country in countries]
Name = Enum("Name", NAME)  # type:ignore

flag_mapping = get_country_flags()
FLAG = [(flag, flag) for code, flag in flag_mapping.items()]
Flag = Enum("Flag", FLAG)  # type:ignore
