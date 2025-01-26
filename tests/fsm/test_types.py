import datetime as pydatetime

import pytest

from outlines.fsm.types import python_types_to_regex
from outlines.types import boolean, date, datetime, integer, number, time


@pytest.mark.parametrize(
    "python_type,regex",
    [
        (int, integer),
        (float, number),
        (bool, boolean),
        (pydatetime.date, date),
        (pydatetime.time, time),
        (pydatetime.datetime, datetime),
    ],
)
def test_python_types(python_type, regex):
    test_regex, _ = python_types_to_regex(python_type)
    assert regex.pattern == test_regex
