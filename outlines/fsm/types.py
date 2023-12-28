import datetime
from typing import Any

INTEGER = r"[+-]?(0|[1-9][0-9]*)"
BOOLEAN = "(True|False)"
FLOAT = rf"{INTEGER}(\.[0-9]+)?([eE][+-][0-9]+)?"
DATE = r"(\d{4})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])"
TIME = r"([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"
DATETIME = rf"({DATE})(\s)({TIME})"


def python_types_to_regex(python_type: Any) -> str:
    if python_type == float:
        return FLOAT
    elif python_type == int:
        return INTEGER
    elif python_type == bool:
        return BOOLEAN
    elif python_type == datetime.date:
        return DATE
    elif python_type == datetime.time:
        return TIME
    elif python_type == datetime.datetime:
        return DATETIME
    else:
        raise NotImplementedError(
            f"The Python type {python_type} is not supported. Please open an issue."
        )
