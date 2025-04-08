import pytest

import llama_cpp
from llama_cpp import LogitsProcessorList

from outlines.models.llamacpp import LlamaCppTypeAdapter, LlamaCppTokenizer
from outlines.processors.structured import RegexLogitsProcessor


@pytest.fixture
def adapter():
    return LlamaCppTypeAdapter()


@pytest.fixture
def logits_processor():
    model = llama_cpp.Llama.from_pretrained(
        "M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )
    return RegexLogitsProcessor(
        regex_string=r"[a-z]+",
        tokenizer=LlamaCppTokenizer(model),
        tensor_library_name="numpy",
    )


def test_llamacpp_type_adapter_format_input(adapter):
    # anything else than a string
    with pytest.raises(NotImplementedError):
        adapter.format_input(["Hello, world!"])

    # string case
    assert adapter.format_input("Hello, world!") == "Hello, world!"


def test_llamacpp_type_adapter_format_output_type(adapter, logits_processor):
    formatted = adapter.format_output_type(logits_processor)
    assert isinstance(formatted, LogitsProcessorList)
    assert len(formatted) == 1
    assert isinstance(formatted[0], RegexLogitsProcessor)
