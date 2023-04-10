import pytest

import outlines.text as text
from outlines.text import render, string
from outlines.text.basic import Add
from outlines.text.var import StringConstant, StringVariable


def test_template_text():
    with pytest.raises(NameError):
        render("String ${one}", two="two")

    t = render("Test")
    assert t == "Test"

    t = render("Test ${variable}", variable="string")
    assert t == "Test string"

    t = render("Test ${variable}", variable=1)
    assert t == "Test 1"

    t = render("Test repeated ${variable} ${variable}", variable="string")
    assert t == "Test repeated string string"

    t = render("Test ${one} ${two}", one="1", two="2")
    assert t == "Test 1 2"


def test_template_string_variable():
    variable = string()
    t = render("Test ${variable}", variable=variable)
    assert isinstance(t.owner.op, Add)
    assert isinstance(t.owner.inputs[0], StringConstant)
    assert isinstance(t.owner.inputs[1], StringVariable)
    assert t.owner.inputs[0].value == "Test "

    variable = string()
    t = render("${variable} test", variable=variable)
    assert isinstance(t.owner.op, Add)
    assert isinstance(t.owner.inputs[0], StringVariable)
    assert isinstance(t.owner.inputs[1], StringConstant)
    assert t.owner.inputs[1].value == " test"


def test_template_few_shots():
    wa = string()
    examples = [["here", "there"], ["this", "that"]]
    prompt = render(
        """
        This is a test

        ${wa}

        % for s, t in examples:
        Search: ${s}
        Trap: ${t}
        % endfor
        """,
        wa=wa,
        examples=examples,
    )
    assert isinstance(prompt, StringVariable)


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


def test_not_prompt():
    with pytest.raises(TypeError, match="template"):

        @text.prompt
        def test_empty(variable):
            pass

    with pytest.raises(TypeError, match="template"):

        @text.prompt
        def test_only_code(variable):
            return variable


def test_prompt_few_shots():
    @text.prompt
    def few_shots_tpl(w, examples):
        """This is a test

        ${w}

        % for s, t in examples:
        Search: ${s}
        Trap: ${t}
        % endfor
        """

    prompt = few_shots_tpl("Test", [["a", "b"], ["c", "d"]])
    assert (
        prompt == "This is a test\n\nTest\n\nSearch: a\nTrap: b\nSearch: c\nTrap: d\n"
    )

    prompt = few_shots_tpl(string(), [["a", "b"], ["c", "d"]])
    assert isinstance(prompt, StringVariable)
