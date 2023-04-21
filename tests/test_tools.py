from typing import List

from outlines.tools import Tool, tool


def test_Tool_basic():
    def test_function():
        pass

    fn = Tool(test_function)
    assert fn.name == "test_function"
    assert fn.description is None
    assert fn.source == "def test_function():\n    pass\n"
    assert fn.signature == ""

    def test_function_description():
        """A description."""
        pass

    fn = Tool(test_function_description)
    assert fn.description == "A description."

    def test_function_set_description():
        """A description."""
        pass

    fn = Tool(test_function_description, description="Another")
    assert fn.description == "Another"

    def test_function_signature(one: int, two: List[str], three: float = 1.0):
        pass

    fn = Tool(test_function_signature)
    assert fn.signature == "one: int, two: List[str], three: float = 1.0"

    def test_function_call(one, two=2):
        return one + two

    fn = Tool(test_function_call)
    assert fn(1) == 3
    assert fn(1, 4) == 5
    assert fn(1, two=3) == 4


def test_tool_decorator():
    @tool
    def test_function():
        pass

    assert test_function.name == "test_function"
    assert test_function.description is None

    @tool(description="Another")
    def test_function_description():
        "A description"
        pass

    assert test_function_description.description == "Another"
