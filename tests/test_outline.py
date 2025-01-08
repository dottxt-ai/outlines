from unittest.mock import MagicMock

import pytest

from outlines.outline import Outline


def test_outline_int_output():
    model = MagicMock()
    model.generate.return_value = "6"

    def template(a: int) -> str:
        return f"What is 2 times {a}?"

    fn = Outline(model, template, int)
    result = fn(3)
    assert result == 6


def test_outline_str_output():
    model = MagicMock()
    model.generate.return_value = "'Hello, world!'"

    def template(a: int) -> str:
        return f"Say 'Hello, world!' {a} times"

    fn = Outline(model, template, str)
    result = fn(1)
    assert result == "Hello, world!"


def test_outline_str_input():
    model = MagicMock()
    model.generate.return_value = "'Hi, Mark!'"

    def template(a: str) -> str:
        return f"Say hi to {a}"

    fn = Outline(model, template, str)
    result = fn(1)
    assert result == "Hi, Mark!"


def test_outline_invalid_output():
    model = MagicMock()
    model.generate.return_value = "not a number"

    def template(a: int) -> str:
        return f"What is 2 times {a}?"

    fn = Outline(model, template, int)
    with pytest.raises(ValueError):
        fn(3)


def test_outline_mismatched_output_type():
    model = MagicMock()
    model.generate.return_value = "'Hello, world!'"

    def template(a: int) -> str:
        return f"What is 2 times {a}?"

    fn = Outline(model, template, int)
    with pytest.raises(
        ValueError,
        match="Unable to parse response: 'Hello, world!'",
    ):
        fn(3)
