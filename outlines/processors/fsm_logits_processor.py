import collections
import json
import math
from typing import DefaultDict, Dict, List, Optional, Protocol

import torch
from pydantic import BaseModel

from outlines.fsm.fsm import CFGFSM, FSM, FSMState, RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.models.tokenizer import Tokenizer
from outlines.processors.base import LogitsProcessor


class FSMLogitsProcessor(LogitsProcessor, Protocol):
    """
    Base class for processing logits with an automaton, either FSM or CFGFSM
    FSMLogitsProcessors are stateful and for ONE TIME USE
    """

    tokenizer: Tokenizer
    fsm: FSM
    fsm_state: DefaultDict[int, FSMState]

    def __init__(self, fsm: FSM, tokenizer: Tokenizer):
        self.fsm = fsm
        self.fsm_state: DefaultDict = collections.defaultdict(int)
        super().__init__(tokenizer)

    def __copy__(self):
        return self.__class__(self.tokenizer, self.fsm.copy())

    def process_logits(
        self, input_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        seq_id = hash(tuple(input_ids))

        # set initial state as 0
        # handles case where input_ids passed include prompt tokens
        if not self.fsm_state:
            self.fsm_state[seq_id] = FSMState(0)

        else:
            # apply state transitions
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            self.fsm_state[seq_id] = self.fsm.next_state(
                self.fsm_state[last_seq_id], last_token
            )

        allowed_token_ids = self.fsm.allowed_token_ids(self.fsm_state[seq_id])

        # bias logits with mask
        mask = torch.full(logits.shape[-1:], -math.inf, device=logits.device)
        mask[allowed_token_ids] = 0

        return logits + mask


class RegexLogitsProcessor(FSMLogitsProcessor):
    def __init__(self, regex_string: str, tokenizer: Tokenizer):
        """Compile the FSM that drives the regex-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            An outlines compatible Tokenizer
        """
        fsm = RegexFSM(regex_string, tokenizer)
        super().__init__(fsm, tokenizer)


class JSONLogitsProcessor(RegexLogitsProcessor):
    def __init__(
        self,
        schema: Dict,
        tokenizer: Tokenizer,
        whitespace_pattern: Optional[str] = None,
    ):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate
        tokenizer
            An outlines compatible Tokenizer
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
        super().__init__(regex_string, tokenizer)


class CFGLogitsProcessor(FSMLogitsProcessor):
    def __init__(self, cfg_str: str, tokenizer: Tokenizer):
        """Compile the FSM that drives the CFG-guided generation.

        Parameters
        ----------
        cfg_str
            A string that represents a grammar
        tokenizer
            An outlines compatible Tokenizer
        """
        fsm = CFGFSM(cfg_str, tokenizer)
        super().__init__(fsm, tokenizer)
