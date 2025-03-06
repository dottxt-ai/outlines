import json
from enum import Enum

import pytest
from llama_cpp import Llama
from pydantic import BaseModel

from outlines.models import LlamaCpp
from outlines.processors import RegexLogitsProcessor
from outlines.types.dsl import Regex, python_types_to_terms, to_regex


def test_load_model():
    model = LlamaCpp(
        Llama.from_pretrained(
            repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
        )
    )

    assert isinstance(model, LlamaCpp)
    assert isinstance(model.model, Llama)


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    return LlamaCpp(
        Llama.from_pretrained(
            repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
        )
    )


def test_llamacpp_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_llamacpp_regex(model):
    regex_str = Regex(r"[0-9]").pattern
    logits_processor = RegexLogitsProcessor(regex_str, model.tokenizer)
    result = model.generate("Respond with one word. Not more.", logits_processor)
    assert isinstance(result, str)


def test_llamacpp_json(model):
    class Foo(BaseModel):
        bar: str

    result = model("foo? Respond with one word.", Foo, max_tokens=1000)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


def test_llamacpp_choice(model):
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    result = model("foo?", Foo)
    assert result == "Foo" or result == "Bar"


def test_llamacpp_text_stop(model):
    result = model.generate("Write the letter a.", None, stop="a", max_tokens=100)
    assert "a" not in result


def test_llamacpp_stream_text_stop(model):
    generator = model.stream("Write the letter a.", None, stop="a", max_tokens=100)

    result = next(generator)
    assert isinstance(result, str)
    assert result != "a"


def test_llamacpp_stream_simple(model):
    generator = model.stream("Respond with one word. Not more.", None)

    x = next(generator)
    assert isinstance(x, str)


def test_llamacpp_stream_regex(model):
    generator = model.stream("Respond with one word. Not more.", Regex(r"[0-9]"))

    x = next(generator)
    assert isinstance(x, str)


def test_llamacpp_stream_json(model):
    class Foo(BaseModel):
        bar: int

    generator = model.stream("foo?", Foo)

    x = next(generator)
    assert x == "{"


def test_llamacpp_stream_choice(model):
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    generator = model.stream("foo?", Foo)

    x = next(generator)
    assert isinstance(x, str)
