from enum import Enum

from . import airports, countries, locale
from outlines.types.dsl import (
    Regex,
    json_schema,
    one_or_more,
    optional,
    regex,
    repeat,
    zero_or_more,
    times,
)

# Python types
integer = Regex(r"[+-]?(0|[1-9][0-9]*)")
boolean = Regex("(True|False)")
number = Regex(rf"{integer.pattern}(\.[0-9]+)?([eE][+-][0-9]+)?")
date = Regex(r"(\d{4})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])")
time = Regex(r"([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])")
datetime = Regex(rf"({date.pattern})(\s)({time.pattern})")

# Basic regex types
digit = Regex(r"\d")
char = Regex(r"\w")
newline = Regex(r"(\r\n|\r|\n)")  # Matched new lines on Linux, Windows & MacOS
whitespace = Regex(r"\s")

# Document-specific types
sentence = Regex(r"[A-Z].*\s*[.!?]")
paragraph = Regex(rf"{sentence.pattern}(?:\s+{sentence.pattern})*\n+")


# The following regex is FRC 5322 compliant and was found at:
# https://emailregex.com/
email = Regex(
    r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""
)

# Matches any ISBN number. Note that this is not completely correct as not all
# 10 or 13 digits numbers are valid ISBNs. See https://en.wikipedia.org/wiki/ISBN
# Taken from O'Reilly's Regular Expression Cookbook:
# https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch04s13.html
#
# TODO: The check digit can only be computed by calling a function to compute it dynamically
isbn = Regex(
    r"(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]"
)
