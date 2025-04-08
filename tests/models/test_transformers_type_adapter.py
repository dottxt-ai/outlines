import pytest

import transformers
from transformers import LogitsProcessorList

import outlines
from outlines.models.transformers import TransformersTypeAdapter
from outlines.processors.structured import RegexLogitsProcessor


@pytest.fixture
def adapter():
    return TransformersTypeAdapter()


@pytest.fixture
def logits_processor():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )
    return RegexLogitsProcessor(
        regex_string=r"[a-z]+",
        tokenizer=model.tokenizer,
        tensor_library_name="torch",
    )


def test_transformers_type_adapter_format_input(adapter):
    with pytest.raises(NotImplementedError):
        adapter.format_input(int)

    assert adapter.format_input("Hello, world!") == "Hello, world!"

    assert adapter.format_input(["Hello, world!", "Hello, world!"]) == [
        "Hello, world!",
        "Hello, world!",
    ]


def test_transformers_type_adapter_format_output_type(
    adapter, logits_processor
):
    formatted = adapter.format_output_type(logits_processor)
    assert isinstance(formatted, LogitsProcessorList)
    assert len(formatted) == 1
    assert isinstance(formatted[0], RegexLogitsProcessor)

    formatted = adapter.format_output_type(None)
    assert formatted is None
