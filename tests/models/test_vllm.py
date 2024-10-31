"""Tests for the `vllm` module."""

import pytest
from transformers import SPIECE_UNDERLINE, AutoTokenizer

from outlines.models.vllm import adapt_tokenizer, convert_token_to_string

TEST_MODEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"


def test_adapt_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL, padding_side="left")
    adapted_tokenizer = adapt_tokenizer(tokenizer=tokenizer)
    assert hasattr(adapted_tokenizer, "vocabulary")
    assert hasattr(adapted_tokenizer, "special_tokens")
    assert adapted_tokenizer.convert_token_to_string == convert_token_to_string


@pytest.mark.parametrize(
    "token, expected",
    [
        ("baz", "baz"),
        ("<0x20>", " <0x20>"),
        (SPIECE_UNDERLINE, f" {SPIECE_UNDERLINE}"),
    ],
)
def test_convert_token_to_string(token, expected):
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL, padding_side="left")
    output = convert_token_to_string(token=token, tokenizer=tokenizer)
    assert output == expected
