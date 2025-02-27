import pytest

import jinja2

from outlines.macros import Macro
from outlines.templates import Template
from outlines.models import Model
from outlines.generate import Generator
from typing import Any


@pytest.fixture(scope="session")
def model():
    class MockModel(Model):
        type_adapter = None

        def generate(self, model_input: str, output_type=None, **kwargs) -> Any:
            return model_input

    return MockModel()


def test_macro_initialization(model):
    template = Template.from_str("Test {{ value }}")
    output_type = None
    macro = Macro(model, template, output_type)

    assert macro.generator == Generator(model, output_type)
    assert macro.template == template


def test_macro_template_call(model):
    template = Template.from_str("Test {{ value }}")
    output_type = None
    macro = Macro(model, template, output_type)
    result = macro(value="example")

    assert result == "Test example"


def test_macro_callable_call(model):
    def template(value):
        return f"Test {value}"

    output_type = None
    macro = Macro(model, template, output_type)
    result = macro("example")

    assert result == "Test example"

def test_macro_template_error(model):
    template = Template.from_str("Test {{ value }}")
    output_type = None
    macro = Macro(model, template, output_type)

    with pytest.raises(jinja2.exceptions.UndefinedError):
        macro(foo="bar")
