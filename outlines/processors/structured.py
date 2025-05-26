"""Logits processors for structured generation.
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

from pydantic import BaseModel
from outlines_core.fsm.json_schema import build_regex_from_schema

from outlines.processors.base_logits_processor import (
    OutlinesLogitsProcessor,
    TensorType
)
from outlines.processors.guide import (
    CFGGuide,
    Guide,
    RegexGuide
)
from outlines.types import JsonSchema

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


class GuideLogitsProcessor(OutlinesLogitsProcessor):
    """Bias generation using a guide.

    Attributes
    ----------
    tokenizer
        The outlines tokenizer used to convert tokens to ids.
    guide
        The outlines guide used to bias the logits.
    """
    tokenizer: "Tokenizer"
    guide: Guide
    _guide_states: Dict[int, Any]
    _seq_start_idx: Optional[int]

    def __init__(
        self, tokenizer: "Tokenizer", guide: Guide, tensor_library_name: str
    ):
        """
        Parameters
        ----------
        tokenizer
            The tokenizer used to convert tokens to ids.
        guide
            The `outlines.processors.guide.Guide` that is used to bias the
            logits.
        tensor_library_name
            The name of the library to use to manipulate the tensors.

        """
        super().__init__(tensor_library_name=tensor_library_name)
        self.tokenizer = tokenizer
        self.guide = guide
        self._guide_states = {hash(tuple([])): self.guide.initial_state}
        self._seq_start_idx = None

    def process_logits(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Use the Guide to bias the logits before sampling the next token.

        Parameters
        ----------
        input_ids
            The ids of the tokens of the existing sequences.
        logits
            The logits for the current generation step.

        Returns
        -------
        TensorType
            The biased logits.

        """
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0]) # type: ignore

        sequence_states: List[int] = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids: # type: ignore
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(self.tensor_adapter.to_list(gen_ids)))

            if curr_state_key not in self._guide_states:
                prev_state = self._guide_states[hash(tuple(self.tensor_adapter.to_list(gen_ids[:-1])))]
                curr_state = self.guide.get_next_state(prev_state, self.tensor_adapter.to_scalar(gen_ids[-1]))
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        allowed_tokens_batch = []
        batch_indices = []
        for i, guide_state in enumerate(sequence_states):
            allowed_tokens = self.guide.get_next_instruction(guide_state).tokens
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(
                self.tensor_adapter.full_like(allowed_tokens, i)
            )  # Store batch index for each allowed token

        device = self.tensor_adapter.get_device(logits)
        allowed_tokens_concat = self.tensor_adapter.to_device(
            self.tensor_adapter.concatenate(allowed_tokens_batch),
            device
        )
        batch_indices_concat = self.tensor_adapter.to_device(
            self.tensor_adapter.concatenate(batch_indices),
            device
        )

        mask = self.tensor_adapter.boolean_ones_like(logits)
        mask[batch_indices_concat, allowed_tokens_concat] = False
        logits = self.tensor_adapter.apply_mask(logits, mask, float("-inf"))

        return logits

    def copy(self) -> "GuideLogitsProcessor":
        """Return a copy of the logits processor."""
        return GuideLogitsProcessor(
            tokenizer=self.tokenizer,
            guide=self.guide.copy(),
            tensor_library_name=self.tensor_adapter.library_name
        )


class RegexLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a regular expression."""
    guide: RegexGuide

    def __init__(
        self,
        regex_string: str,
        tokenizer: "Tokenizer",
        tensor_library_name: str,
    ):
        """
        Parameters
        ----------
        regex_string
            A string that represents a regular expression.
        tokenizer
            An Outlines tokenizer.
        tensor_library_name
            The name of the library to use to manipulate the tensors.

        """
        # Build a guide from the regex string and then pass it to the
        # GuideLogitsProcessor superclass.
        guide = RegexGuide.from_regex(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, guide=guide, tensor_library_name=tensor_library_name)


class JSONLogitsProcessor(RegexLogitsProcessor):
    """Bias generation based on a JSON schema."""
    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        tokenizer: "Tokenizer",
        tensor_library_name: str,
        whitespace_pattern: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate.
        tokenizer
            The tokenizer used to convert tokens to ids.
        tensor_library_name
            The name of the library to use to manipulate the tensors.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`.

        """
        # Convert the JSON schema into a regex string and then pass it to the
        # RegexLogitsProcessor superclass.
        schema_str = JsonSchema(schema).schema
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(
            regex_string=regex_string,
            tokenizer=tokenizer,
            tensor_library_name=tensor_library_name
        )


class CFGLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a context-free grammar."""
    guide: CFGGuide

    def __init__(
        self, cfg_str: str, tokenizer: "Tokenizer", tensor_library_name: str
    ):
        """
        Parameters
        ----------
        cfg_str
            A string that represents a grammar.
        tokenizer
            The tokenizer used to convert tokens to ids.
        tensor_library_name
            The name of the library to use to manipulate the tensors.

        """
        # Build a guide from the CFG string and then pass it to the
        # GuideLogitsProcessor superclass.
        cfg_guide = CFGGuide(cfg_string=cfg_str, tokenizer=tokenizer)
        super().__init__(
            tokenizer=tokenizer,
            guide=cfg_guide,
            tensor_library_name=tensor_library_name
        )

    def process_logits(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Same behavior as GuideLogitsProcessor, but uses rejection
        sampling.

        Parameters
        ----------
        input_ids
            The ids of the tokens of the existing sequences.
        logits
            The logits for the current generation step.

        Returns
        -------
        TensorType
            The biased logits.

        """
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0]) # type: ignore

        sequence_states: List = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids: # type: ignore
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(self.tensor_adapter.to_list(gen_ids)))

            if curr_state_key not in self._guide_states: # pragma: no cover
                prev_state = self._guide_states[hash(tuple(self.tensor_adapter.to_list(gen_ids[:-1])))]
                curr_state = self.guide.get_next_state(prev_state, self.tensor_adapter.to_scalar(gen_ids[-1]))
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        mask = self.tensor_adapter.full_like(logits, -math.inf)
        for i, guide_state in enumerate(sequence_states):
            first_legal_token = next(
                self.guide.iter_valid_token_ids(
                    guide_state, self.tensor_adapter.argsort_descending(logits[i]) # type: ignore
                )
            )
            mask[i, [first_legal_token]] = logits[i, [first_legal_token]] # type: ignore

        return mask
