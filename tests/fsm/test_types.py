import datetime

import pytest

from outlines.fsm.types import (
    BOOLEAN,
    DATE,
    DATETIME,
    FLOAT,
    INTEGER,
    TIME,
    python_types_to_regex,
)


@pytest.mark.parametrize(
    "python_type,regex",
    [
        (int, INTEGER),
        (float, FLOAT),
        (bool, BOOLEAN),
        (datetime.date, DATE),
        (datetime.time, TIME),
        (datetime.datetime, DATETIME),
    ],
)
def test_python_types(python_type, regex):
    test_regex, _ = python_types_to_regex(python_type)
    assert regex == test_regex
