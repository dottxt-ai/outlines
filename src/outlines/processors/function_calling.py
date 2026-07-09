"""Logits processor for function calling with tool call delimiters."""

from enum import Enum
from typing import Any, List

import numpy as np

from outlines.backends import get_logits_processor
from outlines.processors.base_logits_processor import (
    OutlinesLogitsProcessor,
    TensorType,
)


class ToolCallPhase(Enum):
    PASSTHROUGH = "passthrough"
    ARGUMENTS = "arguments"
    CLOSING = "closing"


def _resolve_token_id(token: str, vocabulary: dict[str, int]) -> int:
    token_id = vocabulary.get(token)
    if token_id is None:
        raise ValueError(
            f"Token '{token}' not found in vocabulary. "
            f"The model must have '{token}' as a single special token."
        )
    return token_id


def _force_single_token_row(
    logits_row: TensorType, token_id: int, vocab_size: int
) -> TensorType:
    """Mask a single row of logits to -inf except for the specified token."""
    mask_np = np.ones((1, vocab_size), dtype=np.float32)
    mask_np[0, token_id] = 0.0

    try:
        import torch

        if isinstance(logits_row, torch.Tensor):
            logits_2d = logits_row.unsqueeze(0)
            mask = torch.from_numpy(mask_np).bool().to(logits_row.device)
            return torch.masked_fill(logits_2d, mask, float("-inf"))[0]
    except ImportError:
        pass

    try:
        import mlx.core as mx

        if isinstance(logits_row, mx.array):
            mask = mx.array(mask_np[0].astype(bool))
            return mx.where(mask, float("-inf"), logits_row)
    except ImportError:
        pass

    mask = mask_np[0].astype(bool)
    return np.where(mask, float("-inf"), logits_row)


class FunctionCallingLogitsProcessor(OutlinesLogitsProcessor):
    """Logits processor that switches between free generation and
    constrained JSON generation around tool call delimiters.

    During generation, each sequence in the batch independently
    transitions through these phases:
    1. PASSTHROUGH: logits pass through unmodified until the model
       generates the open delimiter token (e.g. <tool_call>).
    2. ARGUMENTS: logits are constrained by an inner processor to
       match a JSON schema derived from the function signatures.
    3. CLOSING: only the close delimiter token is allowed.
    4. Back to PASSTHROUGH — supports multiple tool calls.
    """

    def __init__(
        self,
        open_token: str,
        close_token: str,
        tools_type: Any,
        model: Any,
    ):
        """
        Parameters
        ----------
        open_token
            The token marking the start of a tool call (e.g. "<tool_call>").
            Must be a single special token in the model's vocabulary.
        close_token
            The token marking the end of a tool call (e.g. "</tool_call>").
            Must be a single special token in the model's vocabulary.
        tools_type
            The type describing the callable tools, e.g. a Pydantic model or
            a ``Union`` of Pydantic models. Tool call arguments are constrained
            to this type.
        model
            An instance of an Outlines model.

        """
        self._prototype_processor = get_logits_processor(tools_type, model)

        vocabulary = model.tokenizer.get_vocab()
        self.open_token_id = _resolve_token_id(open_token, vocabulary)
        self.close_token_id = _resolve_token_id(close_token, vocabulary)

        self._tensor_library_name = model.tensor_library_name
        self._phases: List[ToolCallPhase] = []
        self._arguments_starts: List[int] = []
        self._inner_processors: List[OutlinesLogitsProcessor] = []
        self.is_first_token = True

        super().__init__(model.tensor_library_name)

    def _create_inner_processor(self) -> OutlinesLogitsProcessor:
        from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor

        if isinstance(self._prototype_processor, OutlinesCoreLogitsProcessor):
            return OutlinesCoreLogitsProcessor(
                self._prototype_processor.index, self._tensor_library_name
            )
        return self._prototype_processor

    def reset(self) -> None:
        self._phases = []
        self._arguments_starts = []
        self._inner_processors = []
        self.is_first_token = True

    def process_logits(self, input_ids: TensorType, logits: TensorType) -> TensorType:
        batch_size = self.tensor_adapter.shape(input_ids)[0]
        vocab_size = self.tensor_adapter.shape(logits)[1]

        if self.is_first_token:
            self._phases = [ToolCallPhase.PASSTHROUGH] * batch_size
            self._arguments_starts = [0] * batch_size
            self._inner_processors = [None] * batch_size  # type: ignore
            self.is_first_token = False
        else:
            for i in range(batch_size):
                last_token_id = self.tensor_adapter.to_scalar(input_ids[i][-1])
                self._advance(i, last_token_id, input_ids)

        for i in range(batch_size):
            phase = self._phases[i]

            if phase == ToolCallPhase.PASSTHROUGH:
                continue

            elif phase == ToolCallPhase.ARGUMENTS:
                if self._inner_processors[i] is None:
                    self._inner_processors[i] = self._create_inner_processor()

                inner = self._inner_processors[i]
                args_ids = self.tensor_adapter.unsqueeze(
                    input_ids[i][self._arguments_starts[i] :]
                )
                row_logits = self.tensor_adapter.unsqueeze(logits[i])
                result = inner.process_logits(args_ids, row_logits)
                logits[i] = result[0]

                if hasattr(inner, "_guides"):
                    guides = inner._guides
                    if guides and guides[0].is_finished():
                        self._phases[i] = ToolCallPhase.CLOSING

            elif phase == ToolCallPhase.CLOSING:
                logits[i] = _force_single_token_row(
                    logits[i], self.close_token_id, vocab_size
                )

        return logits

    def _advance(self, seq_idx: int, last_token_id: int, input_ids: TensorType) -> None:
        phase = self._phases[seq_idx]

        if phase == ToolCallPhase.PASSTHROUGH:
            if last_token_id == self.open_token_id:
                self._phases[seq_idx] = ToolCallPhase.ARGUMENTS
                seq_len = self.tensor_adapter.shape(input_ids)[1]
                self._arguments_starts[seq_idx] = seq_len
                self._inner_processors[seq_idx] = None

        elif phase == ToolCallPhase.CLOSING:
            if last_token_id == self.close_token_id:
                self._phases[seq_idx] = ToolCallPhase.PASSTHROUGH
