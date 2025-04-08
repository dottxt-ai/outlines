import pytest

import transformers

from outlines.models.transformers import (
    get_llama_tokenizer_types,
    TransformerTokenizer,
)


TEST_MODEL = "erwanf/gpt2-mini"
TEST_MODEL_SEQ2SEQ = "hf-internal-testing/tiny-random-t5"


@pytest.fixture
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(TEST_MODEL)


@pytest.fixture
def tokenizer_no_pad_token_id(tokenizer):
    tokenizer.pad_token_id = None
    return tokenizer


@pytest.fixture
def tokenizer_seq2seq():
    return transformers.AutoTokenizer.from_pretrained(TEST_MODEL_SEQ2SEQ)


@pytest.fixture
def transformer_tokenizer(tokenizer):
    return TransformerTokenizer(tokenizer)


@pytest.fixture
def another_transformer_tokenizer(tokenizer):
    return TransformerTokenizer(tokenizer)


@pytest.fixture
def transformer_tokenizer_seq2seq(tokenizer_seq2seq):
    return TransformerTokenizer(tokenizer_seq2seq)


def test_get_llama_tokenizer_types():
    tokenizer_types = get_llama_tokenizer_types()
    assert tokenizer_types[0] is transformers.models.llama.LlamaTokenizer
    assert tokenizer_types[1] is transformers.models.llama.LlamaTokenizerFast
    assert tokenizer_types[2] is transformers.models.code_llama.CodeLlamaTokenizer
    assert tokenizer_types[3] is transformers.models.code_llama.CodeLlamaTokenizerFast


def test_transformer_tokenizer_init(
    tokenizer,
    tokenizer_no_pad_token_id
):
    # tokenizer with a pad_token_id
    transformer_tokenizer = TransformerTokenizer(tokenizer)
    assert transformer_tokenizer.tokenizer == tokenizer
    assert transformer_tokenizer.eos_token_id == tokenizer.eos_token_id
    assert transformer_tokenizer.pad_token_id == tokenizer.pad_token_id
    assert transformer_tokenizer.special_tokens == set(tokenizer.all_special_tokens)
    assert transformer_tokenizer.vocabulary == tokenizer.get_vocab()

    # tokenizer with no pad_token_id
    transformer_tokenizer_no_pad_token_id = TransformerTokenizer(tokenizer_no_pad_token_id)
    assert transformer_tokenizer_no_pad_token_id.tokenizer == tokenizer_no_pad_token_id
    assert transformer_tokenizer_no_pad_token_id.eos_token_id == tokenizer_no_pad_token_id.eos_token_id
    assert transformer_tokenizer_no_pad_token_id.pad_token_id == tokenizer_no_pad_token_id.eos_token_id
    assert transformer_tokenizer_no_pad_token_id.special_tokens == set(tokenizer_no_pad_token_id.all_special_tokens)
    assert transformer_tokenizer_no_pad_token_id.vocabulary == tokenizer_no_pad_token_id.get_vocab()


def test_transformer_tokenizer_encode(transformer_tokenizer):
    input_ids, attention_mask = transformer_tokenizer.encode("Hello, world!")
    assert input_ids is not None
    assert attention_mask is not None
    assert input_ids.shape == attention_mask.shape


def test_transformer_tokenizer_decode(transformer_tokenizer):
    input_ids, _ = transformer_tokenizer.encode("Hello, world!")
    decoded_text = transformer_tokenizer.decode(input_ids)
    assert isinstance(decoded_text, list)
    assert "Hello, world!" in decoded_text[0]


def test_transformer_tokenizer_convert_token_to_string(transformer_tokenizer):
    # regular
    transformer_tokenizer.is_llama = False
    token = transformer_tokenizer.tokenizer.tokenize("Hello")[0]
    string = transformer_tokenizer.convert_token_to_string(token)
    assert isinstance(string, str)
    assert "Hello" in string

    # is_llama + <0x20>
    transformer_tokenizer.is_llama = True
    string = transformer_tokenizer.convert_token_to_string("<0x20>")
    assert isinstance(string, str)
    assert " " in string


def test_transformer_tokenizer_eq(
    transformer_tokenizer,
    another_transformer_tokenizer,
    transformer_tokenizer_seq2seq,
):
    # different types of object
    assert transformer_tokenizer.__eq__(1) == NotImplemented

    # regular case
    assert transformer_tokenizer == another_transformer_tokenizer
    assert transformer_tokenizer != transformer_tokenizer_seq2seq

    # with model name and kwargs attributes
    transformer_tokenizer.model_name = "foo"
    transformer_tokenizer.kwargs = {"foo": "bar"}
    another_transformer_tokenizer.model_name = "foo"
    another_transformer_tokenizer.kwargs = {"foo": "bar"}
    assert transformer_tokenizer == another_transformer_tokenizer


def test_transformer_tokenizer_hash(
    transformer_tokenizer,
    another_transformer_tokenizer,
    transformer_tokenizer_seq2seq,
):
    assert isinstance(hash(transformer_tokenizer), int)
    assert hash(transformer_tokenizer) == hash(another_transformer_tokenizer)
    assert hash(transformer_tokenizer) != hash(transformer_tokenizer_seq2seq)


def test_transformer_tokenizer_getstate_setstate(
    transformer_tokenizer,
    another_transformer_tokenizer,
):
    state = transformer_tokenizer.__getstate__()
    assert "tokenizer" in state

    another_transformer_tokenizer.__setstate__(state)
    assert another_transformer_tokenizer == transformer_tokenizer
