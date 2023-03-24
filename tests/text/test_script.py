import pytest

from outlines.text import script, string
from outlines.text.basic import Add
from outlines.text.var import StringConstant, StringVariable


def test_template_text():
    with pytest.raises(TypeError):
        script("String ${one}")(two="two")

    string = "Test"
    t = script(string)()
    assert isinstance(t, StringConstant)
    assert t.value == "Test"

    t = script("Test ${variable}")(variable="string")
    assert t.owner.inputs[0].value == "Test "
    assert t.owner.inputs[1].value == "string"

    t = script("Test ${variable}")(variable=1)
    assert t.owner.inputs[0].value == "Test "
    assert t.owner.inputs[1].value == "1"

    t = script("Test repeated ${variable} ${variable}")(variable="string")
    assert isinstance(t.owner.op, Add)
    assert t.owner.inputs[1].value == "string"
    assert isinstance(t.owner.inputs[0].owner.op, Add)
    assert t.owner.inputs[0].owner.inputs[1].value == " "

    t = script("Test ${one} ${two}")(one="1", two="2")
    assert t.owner.inputs[1].value == "2"
    assert t.owner.inputs[0].owner.inputs[0].owner.inputs[1].value == "1"


def test_template_string_variable():
    variable = string()
    t = script("Test ${variable}")(variable=variable)
    assert isinstance(t.owner.op, Add)
    assert isinstance(t.owner.inputs[0], StringConstant)
    assert isinstance(t.owner.inputs[1], StringVariable)
    assert t.owner.inputs[0].value == "Test "

    variable = string()
    t = script("${variable} test")(variable=variable)
    assert isinstance(t.owner.op, Add)
    assert isinstance(t.owner.inputs[0], StringVariable)
    assert isinstance(t.owner.inputs[1], StringConstant)
    assert t.owner.inputs[1].value == " test"
