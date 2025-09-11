import pytest
from outlines.outputs import Output, StreamingOutput, ToolCallOutput, StreamingToolCallOutput


def test_tool_call_output():
    tool_call = ToolCallOutput(
        name="test_tool",
        args={"param": "value"},
        id="call_123"
    )
    assert tool_call.name == "test_tool"
    assert tool_call.args == {"param": "value"}
    assert tool_call.id == "call_123"


def test_streaming_tool_call_output():
    tool_call = StreamingToolCallOutput(
        name="streaming_tool",
        args="partial_args",
        id="stream_456"
    )
    assert tool_call.name == "streaming_tool"
    assert tool_call.args == "partial_args"
    assert tool_call.id == "stream_456"


def test_output():
    tool_calls = [ToolCallOutput(name="test", args={"arg": "value"})]
    output = Output(content="Hello", tool_calls=tool_calls)

    assert output.content == "Hello"
    assert output.tool_calls == tool_calls
    assert str(output) == "Hello"
    assert "Output(" in repr(output)
    assert output + " World" == "Hello World"
    assert "Hi " + output == "Hi Hello"


def test_streaming_output():
    tool_calls = [StreamingToolCallOutput(name="stream", args="partial")]
    output = StreamingOutput(content="Streaming", tool_calls=tool_calls)

    assert output.content == "Streaming"
    assert output.tool_calls == tool_calls
    assert str(output) == "Streaming"
    assert "StreamingOutput(" in repr(output)
    assert output + " content" == "Streaming content"
    assert "Live " + output == "Live Streaming"
