import pytest

from outlines.models.transformers import TransformerTokenizer


@pytest.fixture
def tokenizer():
    return TransformerTokenizer("gpt2")
