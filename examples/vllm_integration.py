import math
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import vllm
from pydantic import BaseModel
from vllm.model_executor.layers.sampler import (
    _SAMPLING_EPS,
    _apply_min_p,
    _apply_penalties,
    _apply_top_p_top_k,
    _build_sampler_output,
    _get_logits,
    _get_logprobs,
    _get_penalties,
    _get_temperatures,
    _get_top_p_top_k_min_p,
    _prune_hidden_states,
    _sample,
)

from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_object


def _patched_apply_logits_processors(
    logits,
    sampling_metadata,
):
    logits_row_idx = 0
    found_logits_processors = False
    for seq_ids, sampling_params in sampling_metadata.seq_groups:
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True
            for seq_id in seq_ids:
                logits_row = logits[logits_row_idx]
                token_ids = sampling_metadata.seq_data[seq_id].output_token_ids
                for logits_processor in logits_processors:
                    logits_row = logits_processor(seq_id, token_ids, logits_row)
                logits[logits_row_idx] = logits_row
                logits_row_idx += 1
        else:
            logits_row_idx += len(seq_ids)
    if found_logits_processors:
        assert logits_row_idx == logits.shape[0]
    return logits


class JSONLogitsProcessor:
    def __init__(self, pydantic_model, llm):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        pydantic_model
            A Pydantic `BaseModel` that encodes the structure we want
            the model to generate.
        llm
            An instance of `vllm.LLM`

        """
        schema = pydantic_model.schema_json()
        regex_str = build_regex_from_object(schema)
        tokenizer = self.adapt_tokenizer(llm.get_tokenizer())

        fsm = RegexFSM(regex_str, tokenizer)
        self.fsm = fsm

    def __call__(
        self, seq_id: int, input_ids: List[int], scores: torch.Tensor
    ) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""

        if len(input_ids) == 0:  # Initialize the fsm states
            self.fsm_state = defaultdict(int)
        else:
            last_token = input_ids[-1]
            self.fsm_state[seq_id] = self.fsm.next_state(
                self.fsm_state[seq_id], last_token
            )

        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state[seq_id])

        mask = torch.full((scores.shape[-1],), -math.inf, device=scores.device)
        mask[allowed_tokens] = 0
        biased_scores = scores + mask

        return biased_scores

    def adapt_tokenizer(self, tokenizer):
        """Adapt vLLM's tokenizer to use to compile the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. In addition we need to handle the missing spaces to
        Llama's tokenizer to be able to compile FSMs for this model.

        """
        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: str) -> str:
            from transformers.file_utils import SPIECE_UNDERLINE

            string = tokenizer.convert_tokens_to_string([token])

            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

            return string

        tokenizer.convert_token_to_string = convert_token_to_string

        return tokenizer


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence, frequency and repetition penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_metadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ):
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, embedding, embedding_bias, self.vocab_size)

        # Apply logits processors (if any).
        logits = _patched_apply_logits_processors(logits, sampling_metadata)
        # Apply presence and frequency penalties.
        presence_penalties, frequency_penalties, repetition_penalties = _get_penalties(
            sampling_metadata
        )
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        assert len(repetition_penalties) == logits.shape[0]
        logits = _apply_penalties(
            logits,
            sampling_metadata,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
        )

        # Apply temperature scaling.
        temperatures = _get_temperatures(sampling_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        top_ps, top_ks, min_ps = _get_top_p_top_k_min_p(
            sampling_metadata, self.vocab_size
        )
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            logits = _apply_top_p_top_k(logits, top_ps, top_ks)

        do_min_p = any(mp > _SAMPLING_EPS for mp in min_ps)
        if do_min_p:
            logits = _apply_min_p(logits, min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        sample_results = _sample(probs, logprobs, sampling_metadata)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results
        )
        return _build_sampler_output(
            sample_results, sampling_metadata, prompt_logprobs, sample_logprobs
        )


vllm.model_executor.layers.sampler.Sampler = Sampler


class User(BaseModel):
    id: int
    name: str


llm = vllm.LLM(model="gpt2")
logits_processor = JSONLogitsProcessor(User, llm)
result = llm.generate(
    ["A prompt", "Another prompt"],
    sampling_params=vllm.SamplingParams(
        max_tokens=100, logits_processors=[logits_processor]
    ),
)
print(result)
