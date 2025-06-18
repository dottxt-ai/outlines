import datetime
from enum import EnumMeta
from typing import Any, Protocol, Tuple, Type

from outlines.types import Regex, boolean as boolean_regex, date as date_regex
from outlines.types import datetime as datetime_regex
from outlines.types import (
    integer as integer_regex,
    number as number_regex,
    time as time_regex,
)


class FormatFunction(Protocol):
    def __call__(self, sequence: str) -> Any: ...


def python_types_to_regex(python_type: Type) -> Tuple[Regex, FormatFunction]:
    # If it is a custom type
    if isinstance(python_type, Regex):
        custom_regex_str = python_type.pattern

        def custom_format_fn(sequence: str) -> str:
            return str(sequence)

        return Regex(custom_regex_str), custom_format_fn

    if isinstance(python_type, EnumMeta):
        values = python_type.__members__.keys()
        enum_regex_str: str = "(" + "|".join(values) + ")"

        def enum_format_fn(sequence: str) -> str:
            return str(sequence)

        return Regex(enum_regex_str), enum_format_fn

    if python_type is float:

        def float_format_fn(sequence: str) -> float:
            return float(sequence)

        return number_regex, float_format_fn
    elif python_type is int:

        def int_format_fn(sequence: str) -> int:
            return int(sequence)

        return integer_regex, int_format_fn
    elif python_type is bool:

        def bool_format_fn(sequence: str) -> bool:
            return bool(sequence)

        return boolean_regex, bool_format_fn
    elif python_type == datetime.date:

        def date_format_fn(sequence: str) -> datetime.date:
            return datetime.datetime.strptime(sequence, "%Y-%m-%d").date()

        return date_regex, date_format_fn
    elif python_type == datetime.time:

        def time_format_fn(sequence: str) -> datetime.time:
            return datetime.datetime.strptime(sequence, "%H:%M:%S").time()

        return time_regex, time_format_fn
    elif python_type == datetime.datetime:

        def datetime_format_fn(sequence: str) -> datetime.datetime:
            return datetime.datetime.strptime(sequence, "%Y-%m-%d %H:%M:%S")

        return datetime_regex, datetime_format_fn
    else:
        raise NotImplementedError(
            f"The Python type {python_type} is not supported. Please open an issue."
        )
