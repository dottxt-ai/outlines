"""An example illustrating parser-based masking."""
import math
import time
from copy import copy

import torch
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)

from outlines.fsm.parsing import PartialLark, PartialPythonIndenter

revision = None
checkpoint = "Salesforce/codegen-350M-mono"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, trust_remote_code=True, revision=revision
).to(device)

parser = PartialLark.open_from_package(
    "tests",
    "partial_python.lark",
    ["text"],
    parser="lalr",
    postlex=PartialPythonIndenter(),
    start="file_input",
)


class ParserLogitsProcessor(LogitsProcessor):
    """Bias invalid token scores according to a running parse state."""

    def __init__(self, parser):
        self.parser = parser
        self.parser_state = parser.parse("")
        self.states_stack = [self.parser_state]
        self.token_seq = None
        self.token_idx = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.token_seq is None:
            self.token_seq = tokenizer.decode(input_ids[0])
            self.token_idx = len(input_ids[0]) - 1
        else:
            self.token_idx += 1
            self.token_seq += tokenizer.decode(input_ids[0][self.token_idx])

        # Process the last sampled token
        lex_state = self.parser_state.lexer.state
        lex_state.text = self.token_seq

        self.parser.parse_from_state(self.parser_state, is_end=False)

        print(f'parsed:"{self.token_seq}"')

        mask = torch.full_like(scores, -math.inf)

        # Determine which tokens in the vocabulary are valid next tokens
        # given the parser state.
        #
        # TODO: This is a very naive and slow approach.  It could be done in
        # parallel, easily memoized/cached, etc., but there are a few other
        # approaches to try first that will dramatically reduce the
        # amount of work needed here.
        t0 = time.perf_counter()
        for test_token, token_id in tokenizer.vocab.items():
            ps = copy(self.parser_state)
            ls = ps.lexer.state
            ls.text = self.token_seq + tokenizer.convert_tokens_to_string([test_token])

            try:
                self.parser.parse_from_state(ps, is_end=False)
                mask[0][token_id] = 0
            except (EOFError, UnexpectedToken, UnexpectedCharacters, DedentError):
                pass

        print(f"next token masking duration: {time.perf_counter() - t0}")

        return scores + mask


set_seed(20399)

input_text = "def "
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

outputs = model.generate(
    inputs,
    max_length=100,
    temperature=0.1,
    logits_processor=LogitsProcessorList([ParserLogitsProcessor(parser)]),
    renormalize_logits=True,
)

print(tokenizer.decode(outputs[0]))
