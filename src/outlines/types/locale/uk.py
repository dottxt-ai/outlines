"""Locale-specific regex patterns for the United Kingdom."""

from outlines.types.dsl import Regex

phone_number = Regex(r"(\+44\s?|0)((7\d{3}|\d{4})[\s-]?\d{6}|(20\d[\s-]?\d{4}[\s-]?\d{4})|\d{3}[\s-]?\d{3}[\s-]?\d{4})")
postcode = Regex(r"[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}")
