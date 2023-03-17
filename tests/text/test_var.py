import pytest

import outlines
from outlines.graph import Variable
from outlines.text.var import StringConstant


def test_cast():
    with pytest.raises(TypeError):
        outlines.text.as_string([])

    with pytest.raises(TypeError):
        outlines.text.as_string(())

    with pytest.raises(TypeError):
        outlines.text.as_string(Variable())

    s = outlines.text.as_string(1)
    assert type(s) == StringConstant
    assert s.value == "1"

    s = outlines.text.as_string(1.3)
    assert type(s) == StringConstant
    assert s.value == "1.3"

    s = outlines.text.as_string("test")
    assert type(s) == StringConstant
    assert s.value == "test"

    s = outlines.text.string()
    outlines.text.as_string(s)
