import math
from typing import List

import torch
import vllm
from pydantic import BaseModel

from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_object


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
        self.fsm_state = 0

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""

        if len(input_ids) != 0:
            last_token = input_ids[-1]
            self.fsm_state = self.fsm.next_state(self.fsm_state, last_token)

        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state)

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


class User(BaseModel):
    id: int
    name: str


llm = vllm.LLM(model="mistalai/Mistal-7B-v0.1")
logits_processor = JSONLogitsProcessor(User, llm)
result = llm.generate(
    "A prompt",
    sampling_params=vllm.SamplingParams(
        max_tokens=100, logits_processors=[logits_processor]
    ),
)
