from pydantic import BaseModel

import outlines
from outlines.function import Function


def test_function_basic():
    @outlines.prompt
    def test_template(text: str):
        """{{ text }}"""

    class Foo(BaseModel):
        id: int

    fn = Function(test_template, "hf-internal-testing/tiny-random-GPTJForCausalLM", Foo)

    assert fn.generator is None

    result = fn("test")
    assert isinstance(result, BaseModel)
