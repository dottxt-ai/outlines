import base64
import os
import tempfile
from typing import Optional

import pytest
from PIL import Image as PILImage
from io import BytesIO
from pydantic import BaseModel, Field

from outlines.inputs import Image
from outlines.templates import (
    Template,
    build_template_from_string,
    Vision,
    get_fn_name,
    get_fn_args,
    get_fn_description,
    get_fn_source,
    get_fn_signature,
    get_schema,
)


def sample_function(x, y=2):
    """This is a sample function."""
    return x + y

def function_with_annotations(x: int, y: str) -> str:
    """Function with annotations."""
    return f"{x} {y}"

def function_with_no_docstring(x, y):
    return x * y

class CallableClass:
    def __call__(self):
        pass

class PydanticClass(BaseModel):
    foo: str


def test_vision_initialization():
    # Create a simple image for testing
    image = PILImage.new("RGB", (10, 10), color="red")
    image.format = "PNG"

    # Initialize the Vision object
    with pytest.deprecated_call():
        vision = Vision(prompt="Test prompt", image=image)

    # Check that the prompt is set correctly
    assert isinstance(vision, list)
    assert len(vision) == 2
    assert vision[0] == "Test prompt"
    assert isinstance(vision[1], Image)

    # Check that the image is encoded correctly
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    expected_image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    assert vision[1].image_str == expected_image_str

    # Check that the image format is set correctly
    assert vision[1].image_format == "image/png"


def test_vision_invalid_image_format():
    # Create an image without a format
    image = PILImage.new("RGB", (10, 10), color="blue")

    # Expect a TypeError when the image format is not set
    with pytest.deprecated_call():
        with pytest.raises(TypeError, match="Could not read the format"):
            Vision(prompt="Test prompt", image=image)


def render(content: str, filters: Optional[dict] = None, **kwargs):
    template = build_template_from_string(content, filters or {})
    return template.render(kwargs)


def test_render():
    tpl = """
    A test string"""
    assert render(tpl) == "A test string"

    tpl = """
    A test string
    """
    assert render(tpl) == "A test string"

    tpl = """
        A test
        Another test
    """
    assert render(tpl) == "A test\nAnother test"

    tpl = """A test
        Another test
    """
    assert render(tpl) == "A test\nAnother test"

    tpl = """
        A test line
            An indented line
    """
    assert render(tpl) == "A test line\n    An indented line"

    tpl = """
        A test line
            An indented line

    """
    assert render(tpl) == "A test line\n    An indented line\n"


def test_render_escaped_linebreak():
    tpl = """
        A long test \
        that we break \
        in several lines
    """
    assert render(tpl) == "A long test that we break in several lines"

    tpl = """
        Break in \
        several lines \
        But respect the indentation
            on line breaks.
        And after everything \
        Goes back to normal
    """
    assert (
        render(tpl)
        == "Break in several lines But respect the indentation\n    on line breaks.\nAnd after everything Goes back to normal"
    )


def test_render_jinja():
    """Make sure that we can use basic Jinja2 syntax, and give examples
    of how we can use it for basic use cases.
    """

    # Notice the newline after the end of the loop
    examples = ["one", "two"]
    prompt = render(
        """
        {% for e in examples %}
        Example: {{e}}
        {% endfor -%}""",
        examples=examples,
    )
    assert prompt == "Example: one\nExample: two\n"

    # We can remove the newline by cloing with -%}
    examples = ["one", "two"]
    prompt = render(
        """
        {% for e in examples %}
        Example: {{e}}
        {% endfor -%}

        Final""",
        examples=examples,
    )
    assert prompt == "Example: one\nExample: two\nFinal"

    # Same for conditionals
    tpl = """
        {% if is_true %}
        true
        {% endif -%}

        final
        """
    assert render(tpl, is_true=True) == "true\nfinal"
    assert render(tpl, is_true=False) == "final"


def test_render_filters():
    def foo(bar: str) -> str:
        """This is a sample function."""
        return bar

    class PydanticClass(BaseModel):
        foo: str = Field(description="bar")

    def custom_filter(x: str) -> str:
        return x.upper()

    # name filter
    tpl = """
    {{ func | name }}
    """
    assert render(tpl, func=foo) == "foo"

    # description filter
    tpl = """
    {{ func | description }}
    """
    assert render(tpl, func=foo) == "This is a sample function."

    # source filter
    tpl = """
    {{ func | source }}
    """
    assert render(tpl, func=foo) == 'def foo(bar: str) -> str:\n    """This is a sample function."""\n    return bar\n'

    # signature filter
    tpl = """
    {{ func | signature }}
    """
    assert render(tpl, func=foo) == "bar: str"

    # args filter
    tpl = """
    {{ func | args }}
    """
    assert render(tpl, func=foo) == "bar: str"

    # schema filter
    tpl = """
    {{ schema | schema }}
    """
    assert render(tpl, schema=PydanticClass) == '{\n  "foo": "bar"\n}'

    # custom filters
    tpl = """
    {{ name | custom_filter }}
    """
    assert render(tpl, {"custom_filter": custom_filter}, name="John") == "JOHN"


@pytest.fixture
def temp_prompt_file():
    test_dir = tempfile.mkdtemp()

    base_template_path = os.path.join(test_dir, "base_template.txt")
    with open(base_template_path, "w") as f:
        f.write(
            """{% block content %}{% endblock %}
"""
        )

    include_file_path = os.path.join(test_dir, "include.txt")
    with open(include_file_path, "w") as f:
        f.write(
            """{% for example in examples %}
- Q: {{ example.question }}
- A: {{ example.answer }}
{% endfor %}
"""
        )

    prompt_file_path = os.path.join(test_dir, "prompt.txt")
    with open(prompt_file_path, "w") as f:
        f.write(
            """{% extends "base_template.txt" %}

{% block content %}
Here is a prompt with examples:

{% include "include.txt" %}

Now please answer the following question:

Q: {{ question }}
A:
{% endblock %}
"""
        )
    yield prompt_file_path


def test_prompt_from_file(temp_prompt_file):
    prompt = Template.from_file(temp_prompt_file)
    examples = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2 + 2?", "answer": "4"},
    ]
    question = "What is the Earth's diameter?"
    rendered = prompt(examples=examples, question=question)
    expected = """Here is a prompt with examples:

- Q: What is the capital of France?
- A: Paris
- Q: What is 2 + 2?
- A: 4

Now please answer the following question:

Q: What is the Earth's diameter?
A:
"""
    assert rendered.strip() == expected.strip()


def test_prompt_from_str():
    content = """
    Hello, {{ name }}!
    """
    prompt = Template.from_string(content)
    assert prompt(name="World") == "Hello, World!"


def test_template_from_str_with_extra_linebreaks():
    content = """
    Hello, {{ name }}!


    """
    template = build_template_from_string(content)
    assert template.render(name="World") == "Hello, World!\n"


def test_get_fn_name():
    with pytest.raises(TypeError):
        get_fn_name(1)
    assert get_fn_name(sample_function) == "sample_function"
    assert get_fn_name(function_with_annotations) == "function_with_annotations"
    no_name_func = lambda x: x
    assert get_fn_name(no_name_func) == "<lambda>"
    assert get_fn_name(CallableClass()) == "CallableClass"


def test_get_fn_args():
    with pytest.raises(TypeError):
        get_fn_args(1)
    assert get_fn_args(sample_function) == "x, y=2"
    assert get_fn_args(function_with_annotations) == "x: int, y: str"


def test_get_fn_description():
    with pytest.raises(TypeError):
        get_fn_description(1)
    assert get_fn_description(sample_function) == "This is a sample function."
    assert get_fn_description(function_with_annotations) == "Function with annotations."
    assert get_fn_description(function_with_no_docstring) == ""


def test_get_fn_source():
    with pytest.raises(TypeError, match="The `source` filter only applies to callables."):
        get_fn_source(1)
    source = (
        'def sample_function(x, y=2):\n'
        '    """This is a sample function."""\n'
        '    return x + y'
    )
    assert get_fn_source(sample_function).strip() == source


def test_get_fn_signature():
    with pytest.raises(TypeError, match="The `source` filter only applies to callables."):
        get_fn_signature(1)
    sample_function_signature = "x, y=2"
    assert get_fn_signature(sample_function) == sample_function_signature
    function_with_annotations_signature = "x: int, y: str"
    assert get_fn_signature(function_with_annotations) == function_with_annotations_signature


def test_get_schema():
    with pytest.raises(NotImplementedError):
        get_schema(1)

    dict_schema = {"foo": "bar"}
    dict_schema_output = get_schema(dict_schema)
    assert dict_schema_output == '{\n  "foo": "bar"\n}'

    pydantic_schema_output = get_schema(PydanticClass)
    assert pydantic_schema_output == '{\n  "foo": "<foo>"\n}'
