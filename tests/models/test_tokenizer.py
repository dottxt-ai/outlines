import pytest

from outlines.models.tokenizer import Tokenizer, _check_hf_chat_template


def test_tokenizer():
    with pytest.raises(TypeError, match="instantiate abstract"):
        Tokenizer()

def test_check_hf_chat_template():
    from transformers import AutoTokenizer

    assert _check_hf_chat_template(AutoTokenizer.from_pretrained("openai-community/gpt2")) is False
    assert _check_hf_chat_template(AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")) is True
