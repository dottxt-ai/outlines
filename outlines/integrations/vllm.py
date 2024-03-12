"""Make vLLM compatible with Outlines' structured generation.

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
from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, List, Optional, Type, Union

import torch
from pydantic import BaseModel

from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import adapt_tokenizer, convert_json_schema_to_str

if TYPE_CHECKING:
    from vllm import LLM


class RegexLogitsProcessor:
    """Bias vLLM generation based on a regular expression.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, regex_string: str, llm: "LLM"):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression.
        llm
            The vLLM model.

        Raises
        ------
        ValueError
            If the provided LLM instance in `RegexLogitsProcessor` neither has a
            `tokenizer` attribute or a `get_tokenizer` method.
        """
        if hasattr(llm, "get_tokenizer"):
            tokenizer = llm.get_tokenizer()
        elif hasattr(llm, "tokenizer"):
            if hasattr(llm.tokenizer, "tokenizer"):
                tokenizer = llm.tokenizer.tokenizer
            else:
                tokenizer = llm.tokenizer
        else:
            raise ValueError(
                "The provided LLM instance in `RegexLogitsProcessor` neither has a "
                "`tokenizer` attribute or a `get_tokenizer` method."
            )
        tokenizer = adapt_tokenizer(tokenizer=tokenizer)
        self.fsm = RegexGuide(regex_string, tokenizer)
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token.

        Parameters
        ----------
        input_ids
            The tokens of the current sentence.
        scores
            The logits of the current sentence.

        Returns
        -------
        torch.Tensor
            The biased logits.
        """
        seq_id = hash(tuple(input_ids))

        # Initialize the FSM state dictionary if the input_ids are empty, as this means
        # that the input_ids are the first tokens of the sequence.
        if len(input_ids) == 0:
            self._fsm_state = defaultdict(int)
        else:
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            self._fsm_state[seq_id] = self.fsm.get_next_state(
                state=self._fsm_state[last_seq_id], token_id=last_token
            )

        allowed_tokens = self.fsm.get_next_instruction(
            state=self._fsm_state[seq_id]
        ).tokens

        mask = torch.full((scores.shape[-1],), -math.inf, device=scores.device)
        mask[allowed_tokens] = 0
        biased_scores = scores + mask

        return biased_scores


class JSONLogitsProcessor(RegexLogitsProcessor):
    """Bias vLLM generation based on a JSON schema.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        llm: "LLM",
        whitespace_pattern: Optional[str] = None,
    ):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate.
        llm
            The vLLM model.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        """
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string=regex_string, llm=llm)
