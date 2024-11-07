"""
 _______________________________
/ Don't want to self-host?       \
\\ Try .json at http://dottxt.co /
 -------------------------------
       \\   ^__^
        \\  (oo)\\_______
            (__)\\       )\\/\
                ||----w |
                ||     ||

Copyright 2024- the Outlines developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import torch
from pydantic import BaseModel

from outlines.fsm.guide import CFGGuide, Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema, convert_json_schema_to_str

from .base_logits_processor import OutlinesLogitsProcessor

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


class GuideLogitsProcessor(OutlinesLogitsProcessor):
    """Bias generation using a finite

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.Guide` which is used to bias the logits.
    """

    tokenizer: "Tokenizer"
    guide: Guide
    _guide_states: Dict[int, Any]
    _seq_start_idx: Optional[int]

    def __init__(self, tokenizer: "Tokenizer", guide: Guide):
        """A Guide-based logits processor.

        Parameters
        ----------
        tokenizer
            The tokenizer used to convert tokens to ids.
        guide
            The `outlines.fsm.Guide. which is used to bias the logits.
        """
        self.tokenizer = tokenizer
        self.guide = guide
        self._guide_states = {hash(tuple([])): self.guide.initial_state}
        self._seq_start_idx = None

    def process_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.Tensor:
        """Use the Guide to bias the logits before sampling the next token.

        Parameters
        ----------
        input_ids
            The input token ids.
        logits
            The logits.

        Returns
        -------
        torch.Tensor
            The biased logits.
        """
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states: List[Any] = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(gen_ids.tolist()))

            if curr_state_key not in self._guide_states:
                prev_state_key = hash(tuple(gen_ids[:-1].tolist()))
                prev_state = self._guide_states.get(
                    prev_state_key, self.guide.initial_state
                )
                curr_state = self.guide.get_next_state(prev_state, gen_ids[-1].item())
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        mask = torch.ones_like(logits, dtype=torch.bool)

        allowed_tokens_batch = []
        batch_indices = []
        for i, guide_state in enumerate(sequence_states):
            instruction = self.guide.get_next_instruction(guide_state)
            if instruction is None:
                continue  # Skip if no instruction is available
            allowed_tokens = instruction.tokens
            if allowed_tokens is None:
                continue  # Skip if no tokens are allowed
            allowed_tokens = allowed_tokens.to(mask.device, non_blocking=True)

            # Filter out invalid token IDs
            allowed_tokens = allowed_tokens[allowed_tokens < logits.size(1)]
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(torch.full_like(allowed_tokens, i))

        if allowed_tokens_batch:
            allowed_tokens_concat = torch.cat(allowed_tokens_batch)
            batch_indices_concat = torch.cat(batch_indices)

            mask[batch_indices_concat, allowed_tokens_concat] = False

        logits = logits.masked_fill(mask, float("-inf"))

        return logits

    def copy(self) -> "GuideLogitsProcessor":
        """Return a copy of the logits processor."""
        return GuideLogitsProcessor(tokenizer=self.tokenizer, guide=self.guide.copy())


class RegexLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a regular expression.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.RegexGuide. which is used to bias the logits.
    """

    def __init__(self, regex_string: str, tokenizer: "Tokenizer"):
        """Compile the RegexGuide that drives the regex-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            An Outlines tokenizer
        """
        guide = RegexGuide.from_regex(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, guide=guide)


class JSONLogitsProcessor(RegexLogitsProcessor):
    """Bias generation based on a JSON schema.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.RegexGuide. which is used to bias the logits.
    """

    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        tokenizer: "Tokenizer",
        whitespace_pattern: Optional[str] = None,
    ):
        """Compile the Guide that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate.
        tokenizer
            The tokenizer used to convert tokens to ids.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        """
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string=regex_string, tokenizer=tokenizer)


class CFGLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a context-free grammar.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.CFGGuide. which is used to bias the logits.
    """

    guide: CFGGuide

    def __init__(self, cfg_str: str, tokenizer: "Tokenizer"):
        """Compile the CFGGuide that drives the CFG-guided generation.

        Parameters
        ----------
        cfg_str
            A string that represents a grammar
        tokenizer
            The tokenizer used to convert tokens to ids.
        """
        cfg_guide = CFGGuide(cfg_string=cfg_str, tokenizer=tokenizer)
        super().__init__(tokenizer=tokenizer, guide=cfg_guide)

    def process_logits(
        self, input_ids: torch.LongTensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Same behavior as GuideLogitsProcessor, but uses rejection sampling"""
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states: List[Any] = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(gen_ids.tolist()))

            if curr_state_key not in self._guide_states:
                prev_state_key = hash(tuple(gen_ids[:-1].tolist()))
                prev_state = self._guide_states.get(
                    prev_state_key, self.guide.initial_state
                )
                curr_state = self.guide.get_next_state(prev_state, gen_ids[-1].item())
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        mask = torch.full_like(logits, -math.inf)
        for i, guide_state in enumerate(sequence_states):
            valid_tokens = list(
                self.guide.iter_valid_token_ids(
                    guide_state, torch.arange(logits.size(1), device=logits.device)
                )
            )
            if valid_tokens:
                # Keep only valid tokens
                mask[i, valid_tokens] = logits[i, valid_tokens]
            else:
                # No valid tokens; generation should stop
                mask[i] = logits[i]

        return mask
