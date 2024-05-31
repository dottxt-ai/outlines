from transformers import AutoTokenizer

import outlines.caching
from outlines.fsm.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer


def clear_outlines_cache():
    outlines.caching.clear_cache()


def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return TransformerTokenizer(tokenizer)


def ensure_numba_compiled(tokenizer):
    RegexGuide("a", tokenizer)
    return True
