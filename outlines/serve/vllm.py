"""Make vLLM compatible with Outlines' guided generation."""
import json
import math
from collections import defaultdict
from typing import DefaultDict, List, Callable

import torch
from vllm import LLMEngine

from outlines.fsm.fsm import RegexFSM, CFGFSM, FSM
from outlines.fsm.json_schema import build_regex_from_object


def _patched_apply_logits_processors(
    logits,
    sampling_metadata,
):
    """Patch vLLM's logit processor.

    We need to patch the logits processor to pass the `seq_id` so we can
    handle several sequences in `JSONLogitsProcessor`
    """
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


def _adapt_tokenizer(tokenizer):
    """Adapt vLLM's tokenizer to use to compile the FSM.

    The API of Outlines tokenizers is slightly different to that of
    `transformers`. The decoder of outlines, returns a list whereas
    the decode of vLLM returns an str. To sync the vLLM decoder with
    outlines internal api, the decoder should be adapted. In addition
    we need to handle the missing spaces to Llama's tokenizer to be
    able to compile FSMs for this model.

    """
    if getattr(tokenizer, "_outlines_adapted", False):
        return tokenizer

    tokenizer.vocabulary = tokenizer.get_vocab()
    tokenizer.special_tokens = set(tokenizer.all_special_tokens)

    def convert_token_to_string(token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
            return " " + string

        return string

    def change_decoder(
        decoder: Callable[[List[int]], str]
    ) -> Callable[[List[int]], List[str]]:
        """Sync vLLM's decoder with the outlines expectations by returning list"""

        def new_decoder(inp_tokens: List[int]) -> List[str]:
            return [decoder(inp_tokens)]

        return new_decoder

    tokenizer.convert_token_to_string = convert_token_to_string
    tokenizer.decode = change_decoder(tokenizer.decode)
    setattr(tokenizer, "_outlines_adapted", True)

    return tokenizer


class FSMLogitsProcessor:
    def __init__(self):
        fsm = FSM()
        self.fsm = fsm

    def __call__(
        self, seq_id: int, input_ids: List[int], scores: torch.Tensor
    ) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""

        if len(input_ids) == 0:  # Initialize the fsm states
            self.fsm_state: DefaultDict[int, int] = defaultdict(int)
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


class RegexLogitsProcessor(FSMLogitsProcessor):
    def __init__(self, regex_string, llm: LLMEngine):
        """Compile the FSM that drives the regex-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        llm
            An instance of `vllm.LLMEngine`

        """
        adapted_tokenizer = _adapt_tokenizer(llm.tokenizer)
        fsm = RegexFSM(regex_string, adapted_tokenizer)
        self.fsm = fsm


class CFGLogitsProcessor(FSMLogitsProcessor):
    def __init__(self, cfg_string, llm: LLMEngine):
        """Compile the FSM that drives the cfg-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        llm
            An instance of `vllm.LLMEngine`

        """
        adapted_tokenizer = _adapt_tokenizer(llm.tokenizer)
        fsm = CFGFSM(cfg_string, adapted_tokenizer)
        self.fsm = fsm


class JSONLogitsProcessor(RegexLogitsProcessor):
    def __init__(self, schema, llm: LLMEngine):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate
        llm
            An instance of `vllm.LLMEngine`

        """
        if isinstance(schema, dict):
            schema = json.dumps(schema)
        regex_string = build_regex_from_object(schema)
        super().__init__(regex_string, llm)
