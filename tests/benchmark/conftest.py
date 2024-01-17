import pytest

from outlines.fsm.fsm import RegexFSM
from outlines.models.transformers import TransformerTokenizer


@pytest.fixture
def tokenizer():
    return TransformerTokenizer("gpt2")


@pytest.fixture
def ensure_numba_compiled(tokenizer):
    RegexFSM("a", tokenizer)
    return True
