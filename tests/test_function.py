import pytest

import outlines


def test_function_no_types():
    with pytest.raises(TypeError, match="input types"):

        @outlines.fn
        def constant(inp):
            return "constant"

        constant("")

    with pytest.raises(TypeError, match="only supports string arguments"):

        @outlines.fn
        def constant(inp: float):
            return "constant"

        constant("")

    with pytest.raises(TypeError, match="output types"):

        @outlines.fn
        def constant(inp: str):
            return "constant"

        constant("")

    with pytest.raises(TypeError, match="only supports string return types"):

        @outlines.fn
        def constant(inp: str) -> float:
            return 1

        constant("")


def test_function_decorator():
    @outlines.fn
    def constant(inp: str) -> str:
        return "constant"

    inp = outlines.text.string()
    out = constant(inp)
    assert str(out.owner.op) == "FromFunctionOp(constant)"
