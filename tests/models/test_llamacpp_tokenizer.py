import pytest
import sys
from unittest.mock import patch

import llama_cpp
import transformers

from outlines.models.llamacpp import LlamaCppTokenizer


@pytest.fixture
def model():
    model = llama_cpp.Llama.from_pretrained(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )
    setattr(
        model.tokenizer_,
        "hf_tokenizer",
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )
    return model


@pytest.fixture
def model_no_hf_tokenizer():
    model = llama_cpp.Llama.from_pretrained(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )
    del model.tokenizer_
    return model


@pytest.fixture
def different_model():
    model = llama_cpp.Llama.from_pretrained(
        "TheBloke/phi-2-GGUF",
        "phi-2.Q4_K_M.gguf",
    )
    return model


@pytest.fixture
def tokenizer(model):
    return LlamaCppTokenizer(model)


@pytest.fixture
def another_tokenizer(model):
    return LlamaCppTokenizer(model)


@pytest.fixture
def tokenizer_no_hf_tokenizer(model_no_hf_tokenizer):
    return LlamaCppTokenizer(model_no_hf_tokenizer)


@pytest.fixture
def different_tokenizer(different_model):
    return LlamaCppTokenizer(different_model)


def test_llama_cpp_tokenizer_init(tokenizer, tokenizer_no_hf_tokenizer):
    # regular case
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None
    assert isinstance(tokenizer.vocabulary, dict)

    # tokenizer with no hf_tokenizer
    assert tokenizer_no_hf_tokenizer.eos_token_id is not None
    assert tokenizer_no_hf_tokenizer.pad_token_id is not None
    assert isinstance(tokenizer_no_hf_tokenizer.vocabulary, dict)


def test_llama_cpp_tokenizer_encode(tokenizer):
    # batch case
    with pytest.raises(NotImplementedError):
        token_ids, attention_mask = tokenizer.encode(["foo", "bar"])

    # regular case
    token_ids, attention_mask = tokenizer.encode("Hello, world!")
    assert token_ids is not None
    assert attention_mask is not None
    assert len(token_ids) == len(attention_mask)


def test_llama_cpp_tokenizer_decode(tokenizer):
    token_ids, _ = tokenizer.encode("Hello, world!")
    decoded_text = tokenizer.decode(token_ids)
    assert isinstance(decoded_text, list)
    assert "".join(decoded_text).strip() == "Hello, world!"


def test_llama_cpp_tokenizer_convert_token_to_string(
    tokenizer,
    tokenizer_no_hf_tokenizer
):
    # with self._hf_tokenizer
    token_str = tokenizer.convert_token_to_string("<0x20>")
    assert isinstance(token_str, str)

    # without self._hf_tokenizer
    token_str = tokenizer_no_hf_tokenizer.convert_token_to_string("<0x20>")
    assert isinstance(token_str, str)


def test_llama_cpp_tokenizer_eq(tokenizer, another_tokenizer, different_tokenizer):
    assert not tokenizer == 1
    assert tokenizer == another_tokenizer
    assert tokenizer != different_tokenizer


def test_llama_cpp_tokenizer_hash(tokenizer, another_tokenizer, different_tokenizer):
    assert isinstance(hash(tokenizer), int)
    assert hash(tokenizer) == hash(another_tokenizer)
    assert hash(tokenizer) != hash(different_tokenizer)


def test_llama_cpp_tokenizer_getstate(tokenizer):
    state = tokenizer.__getstate__()
    assert isinstance(state, tuple)
    assert len(state) == 5
    assert isinstance(state[0], dict)
    assert isinstance(state[1], int)
    assert isinstance(state[2], str)
    assert isinstance(state[3], int)
    assert isinstance(state[4], list)


def test_llama_cpp_tokenizer_setstate(tokenizer):
    with pytest.raises(NotImplementedError):
        tokenizer.__setstate__(None)
