import os
import tempfile
from typing import Dict, List

import pytest
from pydantic import BaseModel, Field

import outlines
from outlines.prompts import Prompt


def render(content: str, **kwargs):
    template = Prompt._template_from_str(content)
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


def test_prompt_basic():
    @outlines.prompt
    def test_tpl(variable):
        """{{variable}} test"""

    assert list(test_tpl.signature.parameters) == ["variable"]

    with pytest.raises(TypeError):
        test_tpl(v="test")

    p = test_tpl("test")
    assert p == "test test"

    p = test_tpl(variable="test")
    assert p == "test test"

    @outlines.prompt
    def test_single_quote_tpl(variable):
        "${variable} test"

    assert list(test_single_quote_tpl.signature.parameters) == ["variable"]

    p = test_tpl("test")
    assert p == "test test"


def test_prompt_kwargs():
    @outlines.prompt
    def test_kwarg_tpl(var, other_var="other"):
        """{{var}} and {{other_var}}"""

    assert list(test_kwarg_tpl.signature.parameters) == ["var", "other_var"]

    p = test_kwarg_tpl("test")
    assert p == "test and other"

    p = test_kwarg_tpl("test", other_var="kwarg")
    assert p == "test and kwarg"

    p = test_kwarg_tpl("test", "test")
    assert p == "test and test"


def test_no_prompt():
    with pytest.raises(TypeError, match="template"):

        @outlines.prompt
        def test_empty(variable):
            pass

    with pytest.raises(TypeError, match="template"):

        @outlines.prompt
        def test_only_code(variable):
            return variable


def test_prompt_function():
    def empty_fn():
        pass

    def with_description():
        """A description.

        But this is ignored.
        """
        pass

    @outlines.prompt
    def name_description_ppt(fn):
        """
        {{fn|name}}: {{fn|description}}
        """

    rendered = name_description_ppt(empty_fn)
    assert rendered == "empty_fn: "

    rendered = name_description_ppt(with_description)
    assert rendered == "with_description: A description."

    def with_signature(one: int, two: List[str], three: float = 1.0):
        pass

    @outlines.prompt
    def name_signature_ppt(fn):
        """
        {{fn|name}}: {{fn|signature}}
        """

    rendered = name_signature_ppt(with_signature)
    assert rendered == "with_signature: one: int, two: List[str], three: float = 1.0"

    def test_function_call(one, two=2):
        return one + two

    @outlines.prompt
    def source_ppt(fn):
        """
        {{fn|source}}
        """

    rendered = source_ppt(test_function_call)
    assert rendered == "def test_function_call(one, two=2):\n    return one + two\n"


def test_prompt_pydantic_response():
    class SimpleResponse(BaseModel):
        one: str = Field(description="a description")
        two: str

    @outlines.prompt
    def source_ppt(model):
        "{{model | schema }}"

    prompt = source_ppt(SimpleResponse)
    assert prompt == '{\n  "one": "a description",\n  "two": "<two>"\n}'

    class NestedResponse(BaseModel):
        answer: str
        thought: SimpleResponse

    prompt = source_ppt(NestedResponse)
    assert (
        prompt
        == '{\n  "answer": "<answer>",\n  "thought": {\n    "one": "a description",\n    "two": "<two>"\n  }\n}'
    )

    class ConvolutedResponse(BaseModel):
        part_one: NestedResponse
        part_two: SimpleResponse

    prompt = source_ppt(ConvolutedResponse)
    assert (
        prompt
        == '{\n  "part_one": {\n    "answer": "<answer>",\n    "thought": {\n      "one": "a description",\n      "two": "<two>"\n    }\n  },\n  "part_two": {\n    "one": "a description",\n    "two": "<two>"\n  }\n}'
    )


def test_prompt_dict_response():
    response = {"one": "a description", "two": ""}

    @outlines.prompt
    def source_ppt(model):
        "{{model | schema }}"

    prompt = source_ppt(response)
    assert prompt == '{\n  "one": "a description",\n  "two": ""\n}'


def test_prompt_args():
    def no_args():
        pass

    def with_args(x, y, z):
        pass

    def with_annotations(x: bool, y: str, z: Dict[int, List[str]]):
        pass

    def with_defaults(x=True, y="Hi", z={4: ["I", "love", "outlines"]}):
        pass

    def with_annotations_and_defaults(
        x: bool = True,
        y: str = "Hi",
        z: Dict[int, List[str]] = {4: ["I", "love", "outlines"]},
    ):
        pass

    def with_all(
        x1,
        y1,
        z1,
        x2: bool,
        y2: str,
        z2: Dict[int, List[str]],
        x3=True,
        y3="Hi",
        z3={4: ["I", "love", "outlines"]},
        x4: bool = True,
        y4: str = "Hi",
        z4: Dict[int, List[str]] = {4: ["I", "love", "outlines"]},
    ):
        pass

    @outlines.prompt
    def args_prompt(fn):
        """args: {{ fn | args }}"""

    assert args_prompt(no_args) == "args: "
    assert args_prompt(with_args) == "args: x, y, z"
    assert (
        args_prompt(with_annotations)
        == "args: x: bool, y: str, z: Dict[int, List[str]]"
    )
    assert (
        args_prompt(with_defaults)
        == "args: x=True, y='Hi', z={4: ['I', 'love', 'outlines']}"
    )
    assert (
        args_prompt(with_annotations_and_defaults)
        == "args: x: bool = True, y: str = 'Hi', z: Dict[int, List[str]] = {4: ['I', 'love', 'outlines']}"
    )
    assert (
        args_prompt(with_all)
        == "args: x1, y1, z1, x2: bool, y2: str, z2: Dict[int, List[str]], x3=True, y3='Hi', z3={4: ['I', 'love', 'outlines']}, x4: bool = True, y4: str = 'Hi', z4: Dict[int, List[str]] = {4: ['I', 'love', 'outlines']}"
    )


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
    prompt = Prompt.from_file(temp_prompt_file)
    assert prompt.signature is None
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
    prompt = Prompt.from_str(content)
    assert prompt.signature is None
    assert prompt(name="World") == "Hello, World!"


def test_template_from_str_with_extra_linebreaks():
    content = """
    Hello, {{ name }}!


    """
    template = Prompt._template_from_str(content)
    assert template.render(name="World") == "Hello, World!\n"
