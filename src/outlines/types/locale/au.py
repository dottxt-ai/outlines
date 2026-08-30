"""Locale-specific regex patterns for Australia."""

from outlines.types.dsl import Regex

phone_number = Regex(r"(\+61\s?|0)([2378][\s.-]?\d{4}[\s.-]?\d{4}|4\d{2}[\s.-]?\d{3}[\s.-]?\d{3})")
postcode = Regex(r"(0[289]\d{2}|[1-9]\d{3})")
