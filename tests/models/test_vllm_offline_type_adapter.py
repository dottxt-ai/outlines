import pytest

import transformers

import outlines
from outlines.models.vllm_offline import VLLMOfflineTypeAdapter
from outlines.processors.structured import RegexLogitsProcessor


@pytest.fixture
def type_adapter():
    return VLLMOfflineTypeAdapter()


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


def test_vllm_offline_type_adapter_format_input(type_adapter):
    assert type_adapter.format_input("foo") == "foo"
    assert type_adapter.format_input(["foo", "bar"]) == ["foo", "bar"]
    with pytest.raises(NotImplementedError):
        type_adapter.format_input({"foo": "bar"})


def test_vllm_offline_type_adapter_format_output_type(type_adapter, logits_processor):
    assert type_adapter.format_output_type(None) == []
    assert type_adapter.format_output_type(logits_processor) == [logits_processor]
