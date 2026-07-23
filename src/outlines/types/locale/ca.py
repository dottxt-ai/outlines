"""Locale-specific regex patterns for Canada."""

from outlines.types.dsl import Regex

phone_number = Regex(r"(\+1\s?)?(\([0-9]{3}\)|[0-9]{3})[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}")
postal_code = Regex(r"[ABCEGHJKLMNPRSTVXY]\d[A-Z]\s?\d[A-Z]\d")
