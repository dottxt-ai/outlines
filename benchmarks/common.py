from transformers import AutoTokenizer

from outlines.fsm.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer


def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return TransformerTokenizer(tokenizer)


def ensure_numba_compiled(tokenizer):
    RegexGuide("a", tokenizer)
    return True
