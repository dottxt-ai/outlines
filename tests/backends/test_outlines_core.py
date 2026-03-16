import re
from unittest.mock import patch

import llama_cpp
import pytest
import transformers
from outlines_core import Index, Vocabulary

import outlines
from outlines.backends.outlines_core import (
    OutlinesCoreBackend,
    OutlinesCoreLogitsProcessor,
)
from tests.backends.test_backends_utils import simulate_model_calling_processor

try:
    import mlx_lm

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def model_transformers():
    return outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )


def model_llamacpp():
    return outlines.from_llamacpp(
        llama_cpp.Llama.from_pretrained(
            repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
            chat_format="qwen",
        )
    )


def model_mlxlm():
    return outlines.from_mlxlm(*mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit"))


@pytest.fixture
def json_schema():
    return (
        '{"type": "object", "properties": {"name": {"type": "string"}, '
        + '"age": {"type": "integer"}}, "required": ["name", "age"], '
        + '"additionalProperties": false}'
    )


@pytest.fixture
def regex():
    return r"[0-9]{3}"


@pytest.fixture
def cfg():
    return """
?start: sum

?sum: product
| sum "+" product   -> add
| sum "-" product   -> sub

?product: atom
| product "*" atom  -> mul
| product "/" atom  -> div

?atom: NUMBER           -> number
| "-" atom         -> neg
| "(" sum ")"

%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""


def test_outlines_core_processor_torch(regex):
    model = model_transformers()
    tokenizer = model.tokenizer
    hf_tokenizer = model.hf_tokenizer
    backend = OutlinesCoreBackend(model)
    index = Index(regex, backend.vocabulary)
    processor = OutlinesCoreLogitsProcessor(index, "torch")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor, "torch", len(tokenizer.get_vocab()), tokenizer.eos_token_id, 2
        )
        assert re.match(regex, hf_tokenizer.decode(input_ids[0]))
        assert re.match(regex, hf_tokenizer.decode(input_ids[1]))


def test_outlines_core_processor_numpy(regex):
    model = model_llamacpp()
    tokenizer = model.tokenizer
    backend = OutlinesCoreBackend(model)
    index = Index(regex, backend.vocabulary)
    processor = OutlinesCoreLogitsProcessor(index, "numpy")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor, "numpy", len(tokenizer.vocabulary), tokenizer.eos_token_id, 2
        )
        assert re.match(regex, tokenizer.decode(input_ids[0])[0])
        assert re.match(regex, tokenizer.decode(input_ids[1])[0])


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_outlines_core_processor_mlx():
    model = model_mlxlm()
    tokenizer = model.mlx_tokenizer
    backend = OutlinesCoreBackend(model)
    index = Index(r"[0-9]{3}", backend.vocabulary)
    processor = OutlinesCoreLogitsProcessor(index, "mlx")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor, "mlx", len(tokenizer.vocabulary), tokenizer.eos_token_id, 2
        )
        assert re.match(regex, tokenizer.decode(input_ids[0]))
        assert re.match(regex, tokenizer.decode(input_ids[1]))


def test_create_vocabulary_preserves_duplicate_token_ids():
    vocab = {
        "hello": 1,
        "world": 2,
        "<0x20>": 3,
        "▁": 4,
    }

    def token_to_str(token):
        if token in ("<0x20>", "▁"):
            return " "
        return token

    vocabulary = OutlinesCoreBackend.create_outlines_core_vocabulary(
        vocab=vocab,
        eos_token_id=0,
        eos_token="hello",
        token_to_str=token_to_str,
    )

    # 4 original IDs - 1 popped (hello) + 1 EOS added by Vocabulary = 4
    assert len(vocabulary) == 4


models = [
    (model_transformers(), "torch"),
    (model_llamacpp(), "numpy"),
]
if HAS_MLX:
    models.append((model_mlxlm(), "mlx"))


@pytest.mark.parametrize("model, tensor_library_name", models)
def test_outlines_core_backend(model, tensor_library_name, json_schema, regex, cfg):
    # initialization
    backend = OutlinesCoreBackend(model)
    assert isinstance(backend.vocabulary, Vocabulary)
    assert backend.tensor_library_name == tensor_library_name

    # json schema
    processor = backend.get_json_schema_logits_processor(json_schema)
    assert isinstance(processor, OutlinesCoreLogitsProcessor)
    generator = outlines.Generator(model, backend="outlines_core", processor=processor)
    response = generator("Hello, how are you?")
    assert "name" in response

    # regex
    processor = backend.get_regex_logits_processor(regex)
    assert isinstance(processor, OutlinesCoreLogitsProcessor)
    generator = outlines.Generator(model, backend="outlines_core", processor=processor)
    response = generator("Hello, how are you?")
    assert len(response) == 3
    assert int(response)

    # cfg
    with pytest.raises(
        NotImplementedError,
        match="Outlines Core does not support context-free grammar.",
    ):
        backend.get_cfg_logits_processor(cfg)

    # batch + multiple generations
    processor = backend.get_json_schema_logits_processor(json_schema)
    generator = outlines.Generator(model, backend="outlines_core", processor=processor)
    for _ in range(2):
        if tensor_library_name == "torch":
            response = generator.batch(
                ["Create a character", "Hello, how are you?"], max_new_tokens=200
            )
            assert len(response) == 2
            for r in response:
                assert r[0] == "{"
                assert "name" in r
        else:
            response = generator("Create a character", max_tokens=20)
            assert response[0] == "{"
            assert "name" in response


def test_create_vocabulary_preserves_distinct_decoded_strings():
    """Test that tokens decoding to distinct strings each get their own entry
    and that eos_token is correctly excluded from the vocabulary.
    """
    vocab = {
        "▁hello": 100,
        "hello": 200,
        "▁world": 300,
        "world": 400,
        "▁the": 500,
        "<eos>": 0,
    }
    eos_token_id = 0
    eos_token = "<eos>"

    # token_to_str strips the leading "▁" (sentencepiece style)
    def token_to_str(token):
        return token.replace("▁", " ") if token.startswith("▁") else token

    # Capture the formatted_vocab dict passed to Vocabulary
    captured = {}

    def mock_vocabulary(eos_id, fmt_vocab):
        captured.update(fmt_vocab)
        return Vocabulary(eos_id, fmt_vocab)

    with patch(
        "outlines.backends.outlines_core.Vocabulary",
        side_effect=mock_vocabulary,
    ):
        OutlinesCoreBackend.create_outlines_core_vocabulary(
            vocab, eos_token_id, eos_token, token_to_str
        )

    # "hello" comes from both "▁hello" (as " hello") and "hello"
    # They decode to different strings, so each should have exactly one ID
    assert sorted(captured[" hello"]) == [100]
    assert sorted(captured["hello"]) == [200]

    # " world" and "world" similarly
    assert sorted(captured[" world"]) == [300]
    assert sorted(captured["world"]) == [400]

    # " the" should have one ID
    assert captured[" the"] == [500]

    # eos_token should have been removed
    assert eos_token not in captured


def test_create_vocabulary_duplicate_decoded_strings():
    """Test that when token_to_str maps multiple tokens to the SAME string,
    all their IDs are accumulated in a single list.

    This is the core bug from issue #1830.
    """
    # Both tokens decode to the exact same string "hi"
    vocab = {
        "▁hi": 10,
        "hi": 20,
        "extra_hi": 30,
        "<eos>": 0,
    }
    eos_token_id = 0
    eos_token = "<eos>"

    # All three non-eos tokens decode to "hi"
    def token_to_str(token):
        return "hi"

    captured = {}

    def mock_vocabulary(eos_id, fmt_vocab):
        captured.update(fmt_vocab)
        return Vocabulary(eos_id, fmt_vocab)

    with patch(
        "outlines.backends.outlines_core.Vocabulary",
        side_effect=mock_vocabulary,
    ):
        OutlinesCoreBackend.create_outlines_core_vocabulary(
            vocab, eos_token_id, eos_token, token_to_str
        )

    # All three non-eos tokens map to "hi", so all IDs must be present
    # Before the fix, only the last one (30) would survive
    assert sorted(captured["hi"]) == [10, 20, 30]
