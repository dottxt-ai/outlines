from outlines.graph import Apply
from outlines.text.basic import Add, add
from outlines.text.var import StringVariable, string


def test_add_symbolic():
    s, t = string(), string()
    w = add(s, t)
    assert isinstance(w, StringVariable)
    assert isinstance(w.owner, Apply)
    assert isinstance(w.owner.op, Add)
    assert len(w.owner.inputs) == 2
    assert len(w.owner.outputs) == 1

    a = Add()
    assert a.perform(("a", "string"))[0] == "astring"

    w = s + t
    assert isinstance(w, StringVariable)
    assert isinstance(w.owner, Apply)
    assert isinstance(w.owner.op, Add)
    assert len(w.owner.inputs) == 2
    assert len(w.owner.outputs) == 1


def test_add_mixed():
    s, t = "a string", string()
    w = s + t
    assert isinstance(w, StringVariable)
    assert isinstance(w.owner, Apply)
    assert isinstance(w.owner.op, Add)
    assert len(w.owner.inputs) == 2
    assert len(w.owner.outputs) == 1

    s, t = string(), "a string"
    w = s + t
    assert isinstance(w, StringVariable)
    assert isinstance(w.owner, Apply)
    assert isinstance(w.owner.op, Add)
    assert len(w.owner.inputs) == 2
    assert len(w.owner.outputs) == 1
