import pytest

from outlines.graph import Apply, Op, Variable, io_toposort


class MyVar(Variable):
    def __init__(self, value):
        self.value = value
        super().__init__()

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

    def __hash__(self):
        return hash((type(self), self.value))


class MyOp(Op):
    def make_node(self, *inputs):
        result = sum(input.value for input in inputs)
        outputs = [MyVar(result)]
        return Apply(self, inputs, outputs)


op = MyOp()


def test_apply_wrong_args():
    with pytest.raises(TypeError):
        Apply(op, 1.0, [])

    with pytest.raises(TypeError):
        Apply(op, [], 1.0)

    with pytest.raises(TypeError):
        Apply(op, [1.0], [])

    with pytest.raises(TypeError):
        Apply(op, [], [1.0])


def test_Apply():
    i = Variable(name="i")
    o = Variable(name="o")
    a = Apply(op, [i], [o])
    assert len(a.inputs) == 1
    assert len(a.outputs) == 1
    assert a.inputs[0].name == "i"
    assert a.outputs[0].name == "o"
    assert a.outputs[0].owner == a


def test_Apply_multiple_inputs():
    i1, i2 = Variable(name="i1"), Variable(name="i2")
    o = Variable(name="o")
    a = Apply(op, [i1, i2], [o])
    assert len(a.inputs) == 2


def test_Variable_wrong_input():
    owner = "txt"
    with pytest.raises(TypeError):
        Variable(owner)

    owner = Apply(op, [], [])
    index = "i"
    with pytest.raises(TypeError):
        Variable(owner, index)

    owner = Apply(op, [], [])
    index = "i"
    name = 1
    with pytest.raises(TypeError):
        Variable(owner, index, name)


def test_Op():
    v1, v2 = MyVar(1), MyVar(2)
    node = op.make_node(v1, v2)
    assert [x for x in node.inputs] == [v1, v2]
    assert [type(x) for x in node.outputs] == [MyVar]
    assert node.outputs[0].owner is node and node.outputs[0].index == 0


def test_string_formatting():
    v1, v2 = MyVar(1), MyVar(2)
    node = op.make_node(v1, v2)
    assert str(node.op) == "MyOp"
    assert str(v1) == "<MyVar>"
    assert [str(o) for o in node.outputs] == ["MyOp.out"]


def test_toposort_simple():
    r1, r2, r5 = MyVar(1), MyVar(2), MyVar(5)
    o1 = op(r1, r2)
    o1.name = "o1"
    o2 = op(o1, r5)
    o2.name = "o2"

    res = io_toposort([r1, r2, r5], [o2])
    assert res == [o1.owner, o2.owner]
