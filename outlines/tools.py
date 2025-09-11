import sys
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, cast

from pydantic import BaseModel

from outlines.types.dsl import is_callable, is_dict_instance, is_pydantic_model

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import TypedDict
else:  # pragma: no cover
    from typing_extensions import TypedDict


class ToolDef(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    required: list[str]


ToolsInput = List[ToolDef | Callable | BaseModel]


def get_formatted_tools(
    tools: Optional[ToolsInput] = None
) -> Optional[List[ToolDef]]:
    """Convert a ToolsInput into a list of ToolDef instances.

    Parameters
    ----------
    tools : Optional[ToolsInput]
        List of tools to format. Can contain a list of ToolDef, Callable, or
        BaseModel instances.

    Returns
    -------
    Optional[List[ToolDef]]
        List of ToolDef instances. If no tools are provided, returns `None`.

    """
    if not tools:
        return None

    formatted_tools: List[ToolDef] = []

    for tool in tools:
        if is_dict_instance(tool):
            tool_dict = cast(Dict[str, Any], tool)
            if all(
                key in tool_dict
                for key in ["name", "description", "parameters", "required"]
            ):
                formatted_tools.append(cast(ToolDef, tool_dict))
            else:
                missing_keys = (
                    set(tool_dict.keys())
                    - set(["name", "description", "parameters", "required"])
                )
                raise ValueError(
                    f"Invalid ToolDef: {tool}. "
                    + "Expected a dictionary with keys 'name', 'description', "
                    + "'parameters', and 'required'. "
                    + f"Missing keys: {missing_keys}"
                )

        elif is_callable(tool):
            callable_tool = cast(Callable[..., Any], tool)
            formatted_tools.append(_callable_to_tool_def(callable_tool))

        elif is_pydantic_model(tool):
            model_tool = cast(type[BaseModel], tool)
            formatted_tools.append(_pydantic_model_to_tool_def(model_tool))

        else:
            raise ValueError(
                f"Unsupported tool type: {type(tool)}. "
                + "Expected ToolDef, callable, or Pydantic model."
            )

    return formatted_tools


def _callable_to_tool_def(func: Callable) -> ToolDef:
    """Convert a callable to a ToolDef instance.

    Parameters
    ----------
    func : Callable
        The function to convert.

    Returns
    -------
    ToolDef
        ToolDef instance.

    """
    signature = inspect.signature(func)
    name = func.__name__
    description = func.__doc__ or f"Function {name}"
    parameters = {}
    required = []

    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Parameter {param_name} has no annotation. "
                + "All parameters must have an annotation."
            )

        param_type = _type_to_string(param.annotation)
        parameters[param_name] = {
            "type": param_type
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "required": required
    }


def _pydantic_model_to_tool_def(model_class: type[BaseModel]) -> ToolDef:
    """Convert a Pydantic model to a ToolDef instance.

    Parameters
    ----------
    model_class : type[BaseModel]
        The Pydantic model class to convert.

    Returns
    -------
    ToolDef
        ToolDef instance with extracted model information.

    """
    schema = model_class.model_json_schema()
    name = schema.get("title", model_class.__name__)
    description = schema.get("description", f"Model {name}")
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    parameters = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        parameters[prop_name] = {
            "type": prop_type
        }

    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "required": required
    }


def _type_to_string(type_annotation: Any) -> str:
    """Convert a Python type annotation to a string representation.

    Parameters
    ----------
    type_annotation : Any
        The type annotation to convert.

    Returns
    -------
    str
        String representation of the type.

    """
    # Handle Union types (including Optional)
    if (
        hasattr(type_annotation, '__origin__') and
        type_annotation.__origin__ is Union
    ):
        args = type_annotation.__args__
        union_types = []
        for arg in args:
            if arg is type(None):
                union_types.append("null")
            else:
                union_types.append(_type_to_string(arg))
        return "|".join(union_types)

    if type_annotation is str:
        return "string"
    elif type_annotation is int:
        return "integer"
    elif type_annotation is float:
        return "number"
    elif type_annotation is bool:
        return "boolean"
    elif type_annotation is list:
        return "array"
    elif type_annotation is dict:
        return "object"
    else:
        raise ValueError(f"Unsupported type: {type_annotation}")
