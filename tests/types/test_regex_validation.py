import pytest

from outlines.types import Regex


def test_regex_rejects_invalid_pattern():
    """An invalid pattern must fail at construction, naming the user pattern."""
    with pytest.raises(ValueError, match="invalid regular expression"):
        Regex("[a-")


def test_regex_keeps_valid_patterns():
    assert Regex("[a-z]+").matches("abc")
