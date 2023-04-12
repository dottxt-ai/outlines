import pytest

import outlines.text as text


def test_render():
    tpl = """
    A test string"""
    assert text.render(tpl) == "A test string"

    tpl = """
    A test string
    """
    assert text.render(tpl) == "A test string"

    tpl = """
        A test
        Another test
    """
    assert text.render(tpl) == "A test\nAnother test"

    tpl = """A test
        Another test
    """
    assert text.render(tpl) == "A test\nAnother test"

    tpl = """
        A test line
            An indented line
    """
    assert text.render(tpl) == "A test line\n    An indented line"


@pytest.mark.xfail(
    reason="We need a regexp that can match whitespace sequences except those that follow a linebreak"
)
def test_render_escaped_linebreak():
    tpl = """
        A long test \
        that we break \
        in several lines
    """
    assert text.render(tpl) == "A long test that we break in several lines"


def test_render_jinja():
    """Make sure that we can use basic Jinja2 syntax, and give examples
    of how we can use it for basic use cases.
    """

    # Notice the newline after the end of the loop
    examples = ["one", "two"]
    prompt = text.render(
        """
        {% for e in examples %}
        Example: {{e}}
        {% endfor -%}""",
        examples=examples,
    )
    assert prompt == "Example: one\nExample: two\n"

    # We can remove the newline by cloing with -%}
    examples = ["one", "two"]
    prompt = text.render(
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
    assert text.render(tpl, is_true=True) == "true\nfinal"
    assert text.render(tpl, is_true=False) == "final"


def test_prompt_basic():
    @text.prompt
    def test_tpl(variable):
        """{{variable}} test"""

    with pytest.raises(TypeError):
        test_tpl(v="test")

    p = test_tpl("test")
    assert p == "test test"

    p = test_tpl(variable="test")
    assert p == "test test"

    @text.prompt
    def test_single_quote_tpl(variable):
        "${variable} test"

    p = test_tpl("test")
    assert p == "test test"


def test_prompt_kwargs():
    @text.prompt
    def test_kwarg_tpl(var, other_var="other"):
        """{{var}} and {{other_var}}"""

    p = test_kwarg_tpl("test")
    assert p == "test and other"

    p = test_kwarg_tpl("test", other_var="kwarg")
    assert p == "test and kwarg"

    p = test_kwarg_tpl("test", "test")
    assert p == "test and test"


def test_no_prompt():
    with pytest.raises(TypeError, match="template"):

        @text.prompt
        def test_empty(variable):
            pass

    with pytest.raises(TypeError, match="template"):

        @text.prompt
        def test_only_code(variable):
            return variable
