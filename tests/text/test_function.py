import json

from pydantic import BaseModel

import outlines.text as text


def test_function_no_validator():
    def passthrough_model(prompt: str):
        return prompt

    @text.prompt
    def prompt(query: str):
        "{{query}}"

    fn = text.function(passthrough_model, prompt)
    assert fn("Hello") == "Hello"
    assert fn(["Hello", "Hi!"]) == ["Hello", "Hi!"]


def test_function_fn_validator():
    def constant_model(_):
        return "[1, 2, 3]"

    @text.prompt
    def prompt(query: str):
        "{{query}}"

    def validator(result):
        return json.loads(result)

    fn = text.function(constant_model, prompt, validator)
    assert fn("Hello") == [1, 2, 3]
    assert fn(["Hello", "Hi!"]) == [[1, 2, 3], [1, 2, 3]]


def test_function_pydantic_validator():
    class Response(BaseModel):
        thought: str
        command: str

    def constant_model(_):
        return '{"thought": "test thought", "command": "resume"}'

    @text.prompt
    def prompt(query: str):
        "{{query}}"

    fn = text.function(constant_model, prompt, Response)
    result = fn("Hello")
    assert isinstance(result, Response)
    assert result.thought == "test thought"
    assert result.command == "resume"

    result = fn(["Hello", "Hi!"])
    assert isinstance(result, list)
    for resp in result:
        assert isinstance(resp, Response)
        assert resp.thought == "test thought"
        assert resp.command == "resume"
