from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ToolCallOutput:
    """Contains the output of a tool call."""
    name: str
    args: dict[str, Any]
    id: Optional[str] = None


@dataclass
class StreamingToolCallOutput:
    """Contains the output of a streaming tool call."""
    name: str
    args: str
    id: Optional[str] = None


@dataclass
class Output:
    """Contains the output of a response from a model."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallOutput]] = None

    def __str__(self) -> str:
        """Return the content as a string."""
        return self.content or ""

    def __repr__(self) -> str:
        """Return a string representation of the Output object."""
        return f"Output(content={self.content}, tool_calls={self.tool_calls})"

    def __add__(self, other) -> str:
        """Support string concatenation with the content."""
        return str(self) + str(other)

    def __radd__(self, other) -> str:
        """Support string concatenation with the content on the right side."""
        return str(other) + str(self)


@dataclass
class StreamingOutput:
    """Contains the output of a streaming response from a model."""
    content: Optional[str] = None
    tool_calls: Optional[List[StreamingToolCallOutput]] = None

    def __str__(self) -> str:
        """Return the content as a string."""
        return self.content or ""

    def __repr__(self) -> str:
        """Return a string representation of the Output object."""
        return f"StreamingOutput(content={self.content}, tool_calls={self.tool_calls})"

    def __add__(self, other) -> str:
        """Support string concatenation with the content."""
        return str(self) + str(other)

    def __radd__(self, other) -> str:
        """Support string concatenation with the content on the right side."""
        return str(other) + str(self)
