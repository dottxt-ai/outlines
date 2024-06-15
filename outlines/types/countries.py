"""Generate valid country codes and names."""
from enum import Enum

import pycountry

ALPHA_2_CODE = [(country.alpha_2, country.alpha_2) for country in pycountry.countries]
Alpha2 = Enum("Alpha_2", ALPHA_2_CODE)  # type:ignore

ALPHA_3_CODE = [(country.alpha_3, country.alpha_3) for country in pycountry.countries]
Alpha3 = Enum("Alpha_2", ALPHA_3_CODE)  # type:ignore

NUMERIC_CODE = [(country.numeric, country.numeric) for country in pycountry.countries]
Numeric = Enum("Numeric_code", NUMERIC_CODE)  # type:ignore

NAME = [(country.name, country.name) for country in pycountry.countries]
Name = Enum("Name", NAME)  # type:ignore

FLAG = [(country.flag, country.flag) for country in pycountry.countries]
Flag = Enum("Flag", FLAG)  # type:ignore
