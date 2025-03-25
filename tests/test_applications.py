import pytest

import jinja2

from outlines.applications import Application
from outlines.templates import Template
from outlines.models import Model
from typing import Any


@pytest.fixture(scope="session")
def model():
    class MockModel(Model):
        type_adapter = None

        def generate(self, model_input: str, output_type=None, **kwargs) -> Any:
            return model_input

        def generate_stream(self, model_input: str, output_type=None, **kwargs) -> Any:
            return model_input

    return MockModel()


@pytest.fixture(scope="session")
def another_model():
    class MockModel(Model):
        type_adapter = None

        def generate(self, model_input: str, output_type=None, **kwargs) -> Any:
            return model_input

        def generate_stream(self, model_input: str, output_type=None, **kwargs) -> Any:
            return model_input

    return MockModel()


def test_application_initialization():
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)

    assert application.template == template
    assert application.output_type == output_type
    assert application.model is None
    assert application.generator is None


def test_application_template_call(model):
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)
    result = application(model, value="example")

    assert result == "Test example"


def test_application_callable_call(model):
    def template(value):
        return f"Test {value}"

    output_type = None
    application = Application(template, output_type)
    result = application(model, value="example")

    assert result == "Test example"


def test_application_template_error(model):
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)

    with pytest.raises(jinja2.exceptions.UndefinedError):
        application(model, foo="bar")


def test_application_generator_reuse(model, another_model):
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)

    application(model, value="example")
    first_generator = application.generator
    first_model = application.model

    application(model, value="example")
    assert application.model == first_model
    assert application.generator == first_generator

    application(another_model, value="example")
    assert application.model == another_model
    assert application.model != first_model
    assert application.generator != first_generator
