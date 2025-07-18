import numpy as np
import pytest
import torch
import transformers

from outlines.processors.llguidance import LLGuidanceLogitsProcessor

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import llguidance.hf


@pytest.fixture
def llg_tokenizer():
    return llguidance.hf.from_tokenizer(
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )


@pytest.fixture
def grammar():
    return (
        '{"grammars": [{ "json_schema": {"type": "object", "properties":'
        + ' {"name": {"type": "string"}, "age": {"type": "integer"}}, "requ'
        + 'ired": ["name", "age"], "additionalProperties": false} }] }'
    )


def test_llguidance_processor_torch(grammar, llg_tokenizer):
    processor = LLGuidanceLogitsProcessor(grammar, llg_tokenizer, "torch")
    logits = torch.randn(2, llg_tokenizer.vocab_size)
    input_ids = torch.randint(0, llg_tokenizer.vocab_size, (2, 10))
    output = processor(input_ids, logits)
    assert output.shape == (2, llg_tokenizer.vocab_size)
    processor(input_ids, logits)


def test_llguidance_processor_numpy(grammar, llg_tokenizer):
    processor = LLGuidanceLogitsProcessor(grammar, llg_tokenizer, "numpy")
    logits = np.random.randn(2, llg_tokenizer.vocab_size)
    input_ids = np.random.randint(0, llg_tokenizer.vocab_size, (2, 10))
    output = processor(input_ids, logits)
    assert output.shape == (2, llg_tokenizer.vocab_size)
    processor(input_ids, logits)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_llguidance_processor_mlx(grammar, llg_tokenizer):
    processor = LLGuidanceLogitsProcessor(grammar, llg_tokenizer, "mlx")
    logits = mx.random.normal((2, llg_tokenizer.vocab_size))
    input_ids = mx.random.randint(0, llg_tokenizer.vocab_size, (2, 10))
    output = processor(input_ids, logits)
    assert output.shape == (2, llg_tokenizer.vocab_size)
    processor(input_ids, logits)


def test_llguidance_processor_tensorflow(grammar, llg_tokenizer):
    with pytest.raises(TypeError):
        LLGuidanceLogitsProcessor(grammar, llg_tokenizer, "tensorflow")


def test_llguidance_processor_jax(grammar, llg_tokenizer):
    with pytest.raises(TypeError):
        LLGuidanceLogitsProcessor(grammar, llg_tokenizer, "jax")
