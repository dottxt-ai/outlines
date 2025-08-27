import sys
from typing import Literal, Optional, List

from outlines.tools import StreamingToolCall, ToolCall

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class Output(TypedDict):
    type: Literal["assistant"] = "assistant"
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]


class StreamingOutput(TypedDict):
    type: Literal["assistant"] = "assistant"
    content: Optional[str]
    tool_calls: Optional[List[StreamingToolCall]]
