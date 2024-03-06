import datetime
from typing import Any, Callable, Tuple

INTEGER = r"[+-]?(0|[1-9][0-9]*)"
BOOLEAN = "(True|False)"
FLOAT = rf"{INTEGER}(\.[0-9]+)?([eE][+-][0-9]+)?"
DATE = r"(\d{4})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])"
TIME = r"([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"
DATETIME = rf"({DATE})(\s)({TIME})"


def python_types_to_regex(python_type: Any) -> Tuple[str, Callable[[str], Any]]:
    if python_type == float:
        float_format_fn = lambda x: float(x)
        return FLOAT, float_format_fn
    elif python_type == int:
        int_format_fn = lambda x: int(x)
        return INTEGER, int_format_fn
    elif python_type == bool:
        bool_format_fn = lambda x: bool(x)
        return BOOLEAN, bool_format_fn
    elif python_type == datetime.date:
        date_format_fn = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date()
        return DATE, date_format_fn
    elif python_type == datetime.time:
        time_format_fn = lambda s: datetime.datetime.strptime(s, "%H:%M:%S").time()
        return TIME, time_format_fn
    elif python_type == datetime.datetime:
        datetime_format_fn = lambda s: datetime.datetime.strptime(
            s, "%Y-%m-%d %H:%M:%S"
        )
        return DATETIME, datetime_format_fn
    else:
        raise NotImplementedError(
            f"The Python type {python_type} is not supported. Please open an issue."
        )
