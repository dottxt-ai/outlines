import datetime
from enum import EnumMeta
from typing import Any, Protocol, Tuple, Type

from typing_extensions import _AnnotatedAlias, get_args

INTEGER = r"[+-]?(0|[1-9][0-9]*)"
BOOLEAN = "(True|False)"
FLOAT = rf"{INTEGER}(\.[0-9]+)?([eE][+-][0-9]+)?"
DATE = r"(\d{4})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])"
TIME = r"([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"
DATETIME = rf"({DATE})(\s)({TIME})"


class FormatFunction(Protocol):
    def __call__(self, sequence: str) -> Any:
        ...


def python_types_to_regex(python_type: Type) -> Tuple[str, FormatFunction]:
    # If it is a custom type
    if isinstance(python_type, _AnnotatedAlias):
        json_schema = get_args(python_type)[1].json_schema
        type_class = get_args(python_type)[0]

        custom_regex_str = json_schema["pattern"]

        def custom_format_fn(sequence: str) -> Any:
            return type_class(sequence)

        return custom_regex_str, custom_format_fn

    if isinstance(python_type, EnumMeta):
        values = python_type.__members__.keys()
        enum_regex_str: str = "(" + "|".join(values) + ")"

        def enum_format_fn(sequence: str) -> str:
            return str(sequence)

        return enum_regex_str, enum_format_fn

    if python_type == float:

        def float_format_fn(sequence: str) -> float:
            return float(sequence)

        return FLOAT, float_format_fn
    elif python_type == int:

        def int_format_fn(sequence: str) -> int:
            return int(sequence)

        return INTEGER, int_format_fn
    elif python_type == bool:

        def bool_format_fn(sequence: str) -> bool:
            return bool(sequence)

        return BOOLEAN, bool_format_fn
    elif python_type == datetime.date:

        def date_format_fn(sequence: str) -> datetime.date:
            return datetime.datetime.strptime(sequence, "%Y-%m-%d").date()

        return DATE, date_format_fn
    elif python_type == datetime.time:

        def time_format_fn(sequence: str) -> datetime.time:
            return datetime.datetime.strptime(sequence, "%H:%M:%S").time()

        return TIME, time_format_fn
    elif python_type == datetime.datetime:

        def datetime_format_fn(sequence: str) -> datetime.datetime:
            return datetime.datetime.strptime(sequence, "%Y-%m-%d %H:%M:%S")

        return DATETIME, datetime_format_fn
    else:
        raise NotImplementedError(
            f"The Python type {python_type} is not supported. Please open an issue."
        )
