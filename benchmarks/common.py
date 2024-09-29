from transformers import AutoTokenizer

from outlines.models.transformers import TransformerTokenizer


def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return TransformerTokenizer(tokenizer)
