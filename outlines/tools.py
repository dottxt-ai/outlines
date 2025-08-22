import sys
from typing import Any, List, Optional, Union, Callable

from pydantic import BaseModel

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class MCPServer(BaseModel):
    url: str


class ToolDef(TypedDict):
    name: str
    description: str
    parameters: dict
    required: list[str]


class ToolCall(TypedDict):
    name: str
    tool_call_id: Optional[str]
    args: dict[str, Any]


ToolsInput = Union[MCPServer, List[ToolDef | Callable]]


def get_formatted_tools(tools: Optional[ToolsInput] = None) -> List[ToolDef]:
    # TODO: Implement this
    return tools
