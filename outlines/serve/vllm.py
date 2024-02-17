"""Make vLLM compatible with Outlines' structured generation."""
import json
import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional

import torch
from pydantic import BaseModel

from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema


class RegexLogitsProcessor:
    def __init__(self, regex_string, llm):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        llm
            An instance of `vllm.LLM`

        """
        tokenizer = self.adapt_tokenizer(llm.tokenizer.tokenizer)

        fsm = RegexFSM(regex_string, tokenizer)
        self.fsm = fsm

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""

        seq_id = hash(tuple(input_ids))

        if len(input_ids) == 0:  # Initialize the fsm states
            self.fsm_state: DefaultDict[int, int] = defaultdict(int)
        else:
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            self.fsm_state[seq_id] = self.fsm.next_state(
                self.fsm_state[last_seq_id], last_token
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


class JSONLogitsProcessor(RegexLogitsProcessor):
    def __init__(self, schema: Dict, llm, whitespace_pattern: Optional[str] = None):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate
        llm
            An instance of `vllm.LLM`
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
            Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`
        """
        if isinstance(schema, type(BaseModel)):
            schema_str = json.dumps(schema.model_json_schema())
        elif isinstance(schema, Dict):
            schema_str = json.dumps(schema)
        elif isinstance(schema, str):
            schema_str = schema
        else:
            raise ValueError(
                f"Cannot parse schema {schema}. The schema must be either "
                + "a Pydantic object, a dictionary or a string that contains the JSON "
                + "Schema specification"
            )
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string, llm)
