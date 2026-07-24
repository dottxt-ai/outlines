"""Locale-specific regex patterns for France."""

from outlines.types.dsl import Regex

phone_number = Regex(r"(\+33\s?|0)[1-9](\s?\d{2}){4}")
postal_code = Regex(r"0[1-9]\d{3}|[1-8]\d{4}|9[0-5]\d{3}")
