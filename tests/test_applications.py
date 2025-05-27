from typing import Any

import jinja2
import pytest
import transformers

from outlines import from_transformers
from outlines.applications import Application
from outlines.templates import Template


@pytest.fixture(scope="session")
def model():
    return from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("gpt2"),
        transformers.AutoTokenizer.from_pretrained("gpt2"),
    )


@pytest.fixture(scope="session")
def another_model():
    return from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("gpt2"),
        transformers.AutoTokenizer.from_pretrained("gpt2"),
    )


def test_application_initialization():
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)

    assert application.template == template
    assert application.output_type == output_type
    assert application.model is None
    assert application.generator is None


def test_application_generator_no_model():
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)

    with pytest.raises(ValueError):
        application(None, {"value": "example"})


def test_application_template_call(model):
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)
    result = application(model, {"value": "example"}, max_new_tokens=10)

    assert isinstance(result, str)


def test_application_callable_call(model):
    def template(value):
        return f"Test {value}"

    output_type = None
    application = Application(template, output_type)
    result = application(model, {"value": "example"}, max_new_tokens=10)

    assert isinstance(result, str)


def test_application_template_error(model):
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)

    with pytest.raises(jinja2.exceptions.UndefinedError):
        application(model, {"foo": "bar"})


def test_application_generator_reuse(model, another_model):
    template = Template.from_string("Test {{ value }}")
    output_type = None
    application = Application(template, output_type)

    application(model, {"value": "example"}, max_new_tokens=10)
    first_generator = application.generator
    first_model = application.model

    application(model, {"value": "example"}, max_new_tokens=10)
    assert application.model == first_model
    assert application.generator == first_generator

    application(another_model, {"value": "example"}, max_new_tokens=10)
    assert application.model == another_model
    assert application.model != first_model
    assert application.generator != first_generator
