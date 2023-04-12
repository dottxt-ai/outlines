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
        A long test \
        that we break \
        in several lines
    """
    assert text.render(tpl) == "A long test that we break in several lines"


@pytest.mark.xfail(reason="The regex used to strip whitespaces is too aggressive")
def test_render_indented():
    tpl = """
        A test line
            An indented line
    """
    assert text.render(tpl) == "A test line\n    An indented line"


@pytest.mark.xfail(reason="Mako adds newlines after for and if blocks")
def test_render_mako():
    """Make sure that we can use basic Mako syntax."""
    examples = ["one", "two"]
    prompt = text.render(
        """
        % for e in examples:
        Example: ${e}
        % endfor
        """,
        examples=examples,
    )
    assert prompt == "Example: one\nExample: two"

    examples = ["one", "two"]
    prompt = text.render(
        """
        % for i, e in enumerate(examples):
        Example ${i}: ${e}
        % endfor
        """,
        examples=examples,
    )
    assert prompt == "Example 0: one\nExample 1: two"

    tpl = """
        % if is_true:
        true
        % endif
        """
    assert text.render(tpl, is_true=True) == "true"
    assert text.render(tpl, is_true=False) == ""


def test_prompt_basic():
    @text.prompt
    def test_tpl(variable):
        """${variable} test"""

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
        """${var} and ${other_var}"""

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
