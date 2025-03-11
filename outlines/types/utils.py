import dataclasses
import datetime
import sys
from enum import Enum, EnumMeta
from typing import (
    Annotated,
    Any,
    Dict,
    Literal,
    List,
    NewType,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import interegular
from genson import SchemaBuilder
from pydantic import BaseModel

if sys.version_info >= (3, 12):
    from typing import _TypedDictMeta  # type: ignore
else:
    from typing_extensions import _TypedDictMeta  # type: ignore


# Type identification


def is_int(value: Any) -> bool:
    return (
        value is int
        or get_origin(value) is int
        or (get_origin(value) is Annotated and get_args(value)[0] is int)
        or (hasattr(value, "__supertype__") and value.__supertype__ is int)
    )


def is_int_instance(value: Any) -> bool:
    return isinstance(value, int)


def is_float(value: Any) -> bool:
    return (
        value is float
        or get_origin(value) is float
        or (get_origin(value) is Annotated and get_args(value)[0] is float)
        or (hasattr(value, "__supertype__") and value.__supertype__ is float)
    )


def is_float_instance(value: Any) -> bool:
    return isinstance(value, float)


def is_str(value: Any) -> bool:
    return (
        value is str
        or get_origin(value) is str
        or (get_origin(value) is Annotated and get_args(value)[0] is str)
        or (hasattr(value, "__supertype__") and value.__supertype__ is str)
    )


def is_str_instance(value: Any) -> bool:
    return isinstance(value, str)


def is_bool(value: Any) -> bool:
    return (
        value is bool
        or get_origin(value) is bool
        or (get_origin(value) is Annotated and get_args(value)[0] is bool)
        or (hasattr(value, "__supertype__") and value.__supertype__ is bool)
    )


def is_dict_instance(value: Any) -> bool:
    return isinstance(value, dict)


def is_datetime(value: Any) -> bool:
    return value is datetime.datetime or get_origin(value) is datetime.datetime


def is_date(value: Any) -> bool:
    return value is datetime.date or get_origin(value) is datetime.date


def is_time(value: Any) -> bool:
    return value is datetime.time or get_origin(value) is datetime.time


def is_native_dict(value: Any) -> bool:
    return value is dict


def is_typing_dict(value: Any) -> bool:
    return get_origin(value) is dict


def is_typing_list(value: Any) -> bool:
    return get_origin(value) is list


def is_typing_tuple(value: Any) -> bool:
    return get_origin(value) is tuple


def is_union(value: Any) -> bool:
    return get_origin(value) is Union


def is_literal(value: Any) -> bool:
    return get_origin(value) is Literal


def is_dataclass(value: Any) -> bool:
    return isinstance(value, type) and dataclasses.is_dataclass(value)


def is_typed_dict(value: Any) -> bool:
    return isinstance(value, _TypedDictMeta)


def is_pydantic_model(value):
    # needed because generic type cannot be used with `issubclass`    # for Python versions < 3.11
    if get_origin(value) is not None:
        return False

    return isinstance(value, type) and issubclass(value, BaseModel)


def is_genson_schema_builder(value: Any) -> bool:
    return isinstance(value, SchemaBuilder)


def is_enum(value: Any) -> bool:
    return isinstance(value, EnumMeta)


def is_callable(value: Any) -> bool:
    return callable(value) and not isinstance(value, type)


def is_interegular_fsm(value: Any) -> bool:
    return isinstance(value, interegular.fsm.FSM)


# Type conversion


def literal_to_enum(value) -> Enum:
    return Enum(value.__name__, [(arg, arg) for arg in get_args(value)])
