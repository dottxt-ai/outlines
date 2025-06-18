import inspect
import json
import warnings
from enum import Enum
from typing import Callable, Type, Union

from pydantic import BaseModel, create_model


def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
    """Convert a JSON schema to a string.

    Parameters
    ----------
    json_schema
        The JSON schema.

    Returns
    -------
    str
        The JSON schema converted to a string.

    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def get_schema_from_signature(fn: Callable) -> dict:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    signature = inspect.signature(fn)
    arguments = {}
    for name, arg in signature.parameters.items():
        if arg.annotation == inspect._empty:
            raise ValueError("Each argument must have a type annotation")
        else:
            arguments[name] = (arg.annotation, ...)

    try:
        fn_name = fn.__name__
    except Exception as e:
        fn_name = "Arguments"
        warnings.warn(
            f"The function name could not be determined. Using default name 'Arguments' instead. For debugging, here is exact error:\n{e}",
            category=UserWarning,
        )
    model = create_model(fn_name, **arguments)

    return model.model_json_schema()


def get_schema_from_enum(myenum: type[Enum]) -> dict:
    if len(myenum) == 0:
        raise ValueError(
            f"Your enum class {myenum.__name__} has 0 members. If you are working with an enum of functions, do not forget to register them as callable (using `partial` for instance)"
        )
    choices = [
        get_schema_from_signature(elt.value.func)
        if callable(elt.value)
        else {"const": elt.value}
        for elt in myenum
    ]
    schema = {"title": myenum.__name__, "oneOf": choices}
    return schema
