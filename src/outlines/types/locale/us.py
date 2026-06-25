"""Locale-specific regex patterns for the United States."""

from outlines.types.dsl import Regex

zip_code = Regex(r"\d{5}(?:-\d{4})?")
phone_number = Regex(r"(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}")
