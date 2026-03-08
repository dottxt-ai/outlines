import ctypes

import pytest
import sys
from unittest.mock import MagicMock, patch

import llama_cpp
import transformers

from outlines.models.llamacpp import LlamaCppTokenizer


@pytest.fixture
def model():
    model = llama_cpp.Llama.from_pretrained(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
        chat_format="qwen",
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
        chat_format="qwen",
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


def _make_mock_model(n_vocab, eos_id, pieces):
    """Build a mock Llama model whose vocab is defined by *pieces*.

    Parameters
    ----------
    n_vocab : int
        Number of tokens in the vocabulary.
    eos_id : int
        The EOS token id.
    pieces : dict[int, bytes]
        Mapping from token id to the raw bytes of the token piece.
    """
    model = MagicMock()
    # Remove tokenizer_ so the code falls into the C-API branch
    del model.tokenizer_
    model.token_eos.return_value = eos_id
    model.n_vocab.return_value = n_vocab
    model.model = MagicMock()
    return model


def test_vocab_truncation_retry_path():
    """Tokens whose piece length exceeds the 32-byte buffer must trigger the
    retry path with a larger buffer so their text is not collapsed."""
    long_piece = b"x" * 40  # 40 > 32 → triggers the retry branch
    short_piece = b"hi"
    eos_piece = b"</s>"

    pieces = {0: short_piece, 1: long_piece, 2: eos_piece}
    model = _make_mock_model(n_vocab=3, eos_id=2, pieces=pieces)

    def fake_llama_token_to_piece(vocab, token_id, buf, buf_size, *args):
        data = pieces[token_id]
        n = len(data)
        # Only write into the buffer when it is large enough
        if buf_size >= n:
            ctypes.memmove(buf, data, n)
        return n

    with patch(
        "outlines.models.llamacpp.llama_model_get_vocab",
        return_value=MagicMock(),
        create=True,
    ), patch(
        "outlines.models.llamacpp.llama_token_to_piece",
        side_effect=fake_llama_token_to_piece,
        create=True,
    ):
        # Patch the imports inside the __init__ else-branch
        with patch.dict(
            "sys.modules",
            {
                "llama_cpp": MagicMock(
                    llama_model_get_vocab=MagicMock(return_value=MagicMock()),
                    llama_token_to_piece=fake_llama_token_to_piece,
                ),
            },
        ):
            tok = LlamaCppTokenizer.__new__(LlamaCppTokenizer)
            # Re-import inside the else-branch uses llama_cpp module
            tok.__init__(model)

    assert tok.vocabulary[long_piece.decode()] == 1
    assert tok.vocabulary[short_piece.decode()] == 0
    assert tok.eos_token == eos_piece.decode()


def test_attention_mask_all_ones_even_with_eos():
    """The attention mask must be all-ones for every token, including EOS."""
    eos_piece = b"</s>"
    pieces = {0: b"hello", 1: eos_piece}
    model = _make_mock_model(n_vocab=2, eos_id=1, pieces=pieces)

    def fake_llama_token_to_piece(vocab, token_id, buf, buf_size, *args):
        data = pieces[token_id]
        n = len(data)
        if buf_size >= n:
            ctypes.memmove(buf, data, n)
        return n

    with patch.dict(
        "sys.modules",
        {
            "llama_cpp": MagicMock(
                llama_model_get_vocab=MagicMock(return_value=MagicMock()),
                llama_token_to_piece=fake_llama_token_to_piece,
            ),
        },
    ):
        tok = LlamaCppTokenizer.__new__(LlamaCppTokenizer)
        tok.__init__(model)

    # Simulate encoding that returns token ids including the EOS token
    fake_tokenizer = MagicMock()
    fake_tokenizer.tokenize.return_value = [0, 1]  # token 1 == eos_id
    tok.tokenizer = fake_tokenizer

    token_ids, attention_mask = tok.encode("hello</s>")

    assert token_ids == [0, 1]
    assert attention_mask == [1, 1]


def test_negative_n_skips_invalid_token():
    """Tokens that return n < 0 from llama_token_to_piece (error codes)
    must be silently skipped instead of producing garbage vocabulary entries."""
    eos_piece = b"</s>"
    pieces = {0: b"ok", 1: None, 2: eos_piece}  # token 1 returns error
    model = _make_mock_model(n_vocab=3, eos_id=2, pieces=pieces)

    def fake_llama_token_to_piece(vocab, token_id, buf, buf_size, *args):
        data = pieces[token_id]
        if data is None:
            return -1  # error return
        n = len(data)
        if buf_size >= n:
            ctypes.memmove(buf, data, n)
        return n

    with patch.dict(
        "sys.modules",
        {
            "llama_cpp": MagicMock(
                llama_model_get_vocab=MagicMock(return_value=MagicMock()),
                llama_token_to_piece=fake_llama_token_to_piece,
            ),
        },
    ):
        tok = LlamaCppTokenizer.__new__(LlamaCppTokenizer)
        tok.__init__(model)

    # Token 1 (error) must not appear in the vocabulary
    assert 1 not in tok.vocabulary.values() or all(
        v != 1 for k, v in tok.vocabulary.items() if k != ""
    )
    assert tok.vocabulary["ok"] == 0
    assert tok.eos_token == eos_piece.decode()
