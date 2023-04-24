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
