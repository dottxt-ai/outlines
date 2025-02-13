import datetime as pydatetime

import pytest

from outlines.fsm.types import python_types_to_regex
from outlines import types


@pytest.mark.parametrize(
    "python_type,custom_type",
    [
        (int, types.integer),
        (float, types.number),
        (bool, types.boolean),
        (pydatetime.date, types.date),
        (pydatetime.time, types.time),
        (pydatetime.datetime, types.datetime),
    ],
)
def test_python_types(python_type, custom_type):
    test_regex, _ = python_types_to_regex(python_type)
    assert custom_type.pattern == test_regex.pattern
