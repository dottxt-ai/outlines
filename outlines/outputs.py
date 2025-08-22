import sys
from typing import Literal, Optional

from outlines.tools import ToolCall

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class Output(TypedDict):
    content: str
    type: Literal["system", "tool_call", "user", "assistant"]
    tool_calls: Optional[list[ToolCall]]
    reasoning: Optional[str]
