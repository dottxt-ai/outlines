"""Tests for the FunctionCallingLogitsProcessor."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from pydantic import BaseModel

from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
from outlines.processors.function_calling import (
    FunctionCallingLogitsProcessor,
    ToolCallPhase,
)

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

libraries = ["numpy", "torch"]
if HAS_MLX:
    libraries.append("mlx")

VOCAB_SIZE = 10
OPEN_TOKEN_ID = 7
CLOSE_TOKEN_ID = 8


class StubInnerProcessor(OutlinesLogitsProcessor):
    """Passes logits through and reports is_finished() after N calls."""

    def __init__(self, tensor_library_name: str, finish_after: int = 3):
        self._call_count = 0
        self._finish_after = finish_after
        self._finished = False
        self._guides = [self]
        super().__init__(tensor_library_name)

    def reset(self) -> None:
        self._call_count = 0
        self._finished = False

    def is_finished(self) -> bool:
        return self._finished

    def process_logits(self, input_ids, logits):
        self._call_count += 1
        if self._call_count >= self._finish_after:
            self._finished = True
        return logits


class DummyToolCall(BaseModel):
    name: str


def make_arrays(library, token_ids, logits_values=None):
    if logits_values is None:
        logits_values = [1.0] * VOCAB_SIZE
    if library == "numpy":
        return (
            np.array([list(token_ids)], dtype=np.int64),
            np.array([logits_values], dtype=np.float32),
        )
    elif library == "torch":
        return (
            torch.tensor([list(token_ids)], dtype=torch.long),
            torch.tensor([logits_values], dtype=torch.float32),
        )
    elif library == "mlx":
        return (
            mx.array([list(token_ids)], dtype=mx.int32),
            mx.array([logits_values], dtype=mx.float32),
        )
    raise ValueError(f"Unsupported library: {library}")


def _stub_factory(library, finish_after):
    """Returns a factory that creates StubInnerProcessors."""

    def factory():
        return StubInnerProcessor(library, finish_after=finish_after)

    return factory


def make_processor(library, finish_after=3):
    model = MagicMock()
    model.tokenizer.get_vocab.return_value = {
        "<tool_call>": OPEN_TOKEN_ID,
        "</tool_call>": CLOSE_TOKEN_ID,
    }
    model.tensor_library_name = library

    factory = _stub_factory(library, finish_after)

    with patch(
        "outlines.processors.function_calling.get_json_schema_logits_processor",
        side_effect=lambda *args, **kwargs: factory(),
    ):
        proc = FunctionCallingLogitsProcessor(
            "<tool_call>", "</tool_call>", DummyToolCall, model
        )

    # Replace _create_inner_processor so subsequent calls also use stubs
    proc._create_inner_processor = factory

    return proc


# --- Unit tests ---


@pytest.mark.parametrize("library", libraries)
def test_passthrough_returns_logits_unmodified(library):
    proc = make_processor(library)
    input_ids, logits = make_arrays(library, [1, 2, 3])
    result = proc(input_ids, logits)
    assert proc._phases[0] == ToolCallPhase.PASSTHROUGH
    if library == "numpy":
        np.testing.assert_array_equal(result, logits)
    elif library == "torch":
        assert torch.equal(result, logits)
    elif library == "mlx":
        assert mx.array_equal(result, logits)


@pytest.mark.parametrize("library", libraries)
def test_phase_transitions_full_cycle(library):
    proc = make_processor(library, finish_after=2)

    proc(*make_arrays(library, [1]))
    assert proc._phases[0] == ToolCallPhase.PASSTHROUGH

    proc(*make_arrays(library, [1, OPEN_TOKEN_ID]))
    assert proc._phases[0] == ToolCallPhase.ARGUMENTS

    proc(*make_arrays(library, [1, OPEN_TOKEN_ID, 2]))
    assert proc._phases[0] == ToolCallPhase.CLOSING

    proc(*make_arrays(library, [1, OPEN_TOKEN_ID, 2, CLOSE_TOKEN_ID]))
    assert proc._phases[0] == ToolCallPhase.PASSTHROUGH


@pytest.mark.parametrize("library", libraries)
def test_arguments_phase_delegates_to_inner(library):
    proc = make_processor(library, finish_after=5)
    proc(*make_arrays(library, [1]))
    proc(*make_arrays(library, [1, OPEN_TOKEN_ID]))
    assert proc._phases[0] == ToolCallPhase.ARGUMENTS
    assert proc._inner_processors[0]._call_count == 1


@pytest.mark.parametrize("library", libraries)
def test_closing_phase_only_allows_close_token(library):
    proc = make_processor(library, finish_after=1)
    proc(*make_arrays(library, [1]))
    proc(*make_arrays(library, [1, OPEN_TOKEN_ID]))
    assert proc._phases[0] == ToolCallPhase.CLOSING

    logits_values = list(range(VOCAB_SIZE))
    result = proc(*make_arrays(library, [1, OPEN_TOKEN_ID, 2], logits_values))

    for token_id in range(VOCAB_SIZE):
        if library == "numpy":
            val = float(result[0][token_id])
        elif library == "torch":
            val = result[0][token_id].item()
        elif library == "mlx":
            val = result[0][token_id].item()

        if token_id == CLOSE_TOKEN_ID:
            assert val == float(logits_values[token_id])
        else:
            assert val == float("-inf")


@pytest.mark.parametrize("library", libraries)
def test_multiple_tool_calls(library):
    proc = make_processor(library, finish_after=2)

    proc(*make_arrays(library, [1]))
    proc(*make_arrays(library, [1, OPEN_TOKEN_ID]))
    assert proc._phases[0] == ToolCallPhase.ARGUMENTS

    proc(*make_arrays(library, [1, OPEN_TOKEN_ID, 2]))
    proc(*make_arrays(library, [1, OPEN_TOKEN_ID, 2, CLOSE_TOKEN_ID]))
    assert proc._phases[0] == ToolCallPhase.PASSTHROUGH

    proc(*make_arrays(library, [1, OPEN_TOKEN_ID, 2, CLOSE_TOKEN_ID, 5]))
    proc(*make_arrays(library, [1, OPEN_TOKEN_ID, 2, CLOSE_TOKEN_ID, 5, OPEN_TOKEN_ID]))
    assert proc._phases[0] == ToolCallPhase.ARGUMENTS


@pytest.mark.parametrize("library", libraries)
def test_reset(library):
    proc = make_processor(library, finish_after=5)
    proc(*make_arrays(library, [1]))
    proc(*make_arrays(library, [1, OPEN_TOKEN_ID]))
    assert proc._phases[0] == ToolCallPhase.ARGUMENTS

    proc.reset()
    assert proc._phases == []
    assert proc.is_first_token is True


@pytest.mark.parametrize("library", ["numpy", "torch"])
def test_batch_sequences_in_different_phases(library):
    proc = make_processor(library, finish_after=5)

    # Batch of 2 sequences: both start in PASSTHROUGH
    if library == "numpy":
        input_ids = np.array([[1, 2], [1, 2]], dtype=np.int64)
        logits = np.ones((2, VOCAB_SIZE), dtype=np.float32)
    elif library == "torch":
        input_ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
        logits = torch.ones((2, VOCAB_SIZE), dtype=torch.float32)

    proc(input_ids, logits)
    assert proc._phases[0] == ToolCallPhase.PASSTHROUGH
    assert proc._phases[1] == ToolCallPhase.PASSTHROUGH

    # Sequence 0 hits open token, sequence 1 stays in passthrough
    if library == "numpy":
        input_ids = np.array([[1, 2, OPEN_TOKEN_ID], [1, 2, 3]], dtype=np.int64)
        logits = np.ones((2, VOCAB_SIZE), dtype=np.float32)
    elif library == "torch":
        input_ids = torch.tensor([[1, 2, OPEN_TOKEN_ID], [1, 2, 3]], dtype=torch.long)
        logits = torch.ones((2, VOCAB_SIZE), dtype=torch.float32)

    result = proc(input_ids, logits)

    assert proc._phases[0] == ToolCallPhase.ARGUMENTS
    assert proc._phases[1] == ToolCallPhase.PASSTHROUGH

    # Sequence 1 logits should be unmodified, sequence 0 should be constrained
    if library == "numpy":
        np.testing.assert_array_equal(result[1], np.ones(VOCAB_SIZE, dtype=np.float32))
    elif library == "torch":
        assert torch.equal(result[1], torch.ones(VOCAB_SIZE, dtype=torch.float32))


# --- Integration test ---


def _simulate_generation(proc, generated_ids, vocab_size, steps):
    for _ in range(steps):
        input_ids = torch.tensor([generated_ids], dtype=torch.long)
        result = proc(input_ids, torch.randn(1, vocab_size))
        generated_ids.append(torch.argmax(result, dim=-1).item())
        if proc._phases[0] == ToolCallPhase.CLOSING:
            break


def test_integration_with_real_fsm():
    """Verify that constrained generation produces valid JSON when
    paired with a real OutlinesCoreLogitsProcessor and tokenizer."""
    import transformers

    import outlines
    from outlines.backends.outlines_core import OutlinesCoreBackend

    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )
    backend = OutlinesCoreBackend(model)
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "name": {"const": "get_weather"},
                "city": {"type": "string", "maxLength": 10},
            },
            "required": ["name", "city"],
            "additionalProperties": False,
        }
    )

    open_id = model.tokenizer.eos_token_id
    close_id = open_id - 1
    vocab_size = len(model.tokenizer.get_vocab())
    inner = backend.get_json_schema_logits_processor(schema)

    mock_model = MagicMock()
    mock_model.tokenizer.get_vocab.return_value = {
        **model.tokenizer.get_vocab(),
        "<tool_call>": open_id,
        "</tool_call>": close_id,
    }
    mock_model.tensor_library_name = "torch"

    with patch(
        "outlines.processors.function_calling.get_json_schema_logits_processor",
        return_value=inner,
    ):
        proc = FunctionCallingLogitsProcessor(
            "<tool_call>", "</tool_call>", DummyToolCall, mock_model
        )

    generated_ids = [100]

    _simulate_generation(proc, generated_ids, vocab_size, 3)
    assert proc._phases[0] == ToolCallPhase.PASSTHROUGH

    generated_ids.append(open_id)
    _simulate_generation(proc, generated_ids, vocab_size, 200)
    assert proc._phases[0] == ToolCallPhase.CLOSING

    open_idx = generated_ids.index(open_id)
    json_tokens = [
        t for t in generated_ids[open_idx + 1 :] if t != model.tokenizer.eos_token_id
    ]
    parsed = json.loads(model.tokenizer.tokenizer.decode(json_tokens))
    assert parsed["name"] == "get_weather"
    assert "city" in parsed
