import json as pyjson
from typing import Callable, Iterator, List, Optional, Union

import torch
from pydantic import BaseModel

from outlines.fsm.fsm import FSMState, RegexFSM, StopAtTokenFSM
from outlines.fsm.json_schema import build_regex_from_object, get_schema_from_signature
from outlines.fsm.types import python_types_to_regex
from outlines.generate.generator import (
    init_generator_state,
    sequence_generator,
    token_generator,
)
from outlines.generate.samplers import Sampler, multinomial


class SequenceGenerator:
    def __init__(self, fsm, model, sampler, device):
        self.generate_token = token_generator(model, sampler)
        self.fsm = fsm
        self.tokenizer = model.tokenizer
        self.device = device

    def format_sequence(self, sequence):
        return sequence

    def __call__(
        self,
        prompt,
        kv_cache: Optional[torch.tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Union[str, List[str]]:
        sequence_generator = self.stream(prompt, kv_cache, rng)
        tokens = [token for token in sequence_generator]
        sequences = [
            self.format_sequence("".join(sequence)) for sequence in list(zip(*tokens))
        ]
        return sequences if len(sequences) > 1 else sequences[0]

    def stream(
        self,
        prompt: str,
        kv_cache: Optional[torch.tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Iterator[Union[List[str], str]]:
        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        init_state = init_generator_state(self.tokenizer, self.device, prompt, kv_cache)

        token_ids = init_state[1]
        num_sequences = token_ids.shape[0]

        init_fsm_states = [FSMState(0) for _ in range(num_sequences)]

        states = sequence_generator(
            self.generate_token, self.fsm, init_state, init_fsm_states, rng
        )

        def token_generator() -> Iterator[Union[List[str], str]]:
            while True:
                try:
                    sequence = next(states)
                except StopIteration:
                    return

                next_token_ids = sequence.token_ids[:, -1]
                next_tokens = self.tokenizer.decode(next_token_ids)

                yield next_tokens

        return token_generator()


def text(model, max_tokens: Optional[int] = None, *, sampler: Sampler = multinomial):
    eos_token = model.tokenizer.eos_token_id
    fsm = StopAtTokenFSM(model.tokenizer, eos_token, max_tokens)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


def regex(
    model,
    regex_str: str,
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial,
):
    fsm = RegexFSM(regex_str, model.tokenizer, max_tokens)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


def format(
    model, python_type, max_tokens: Optional[int] = None, sampler: Sampler = multinomial
):
    regex_str = python_types_to_regex(python_type)
    return regex(model, regex_str, max_tokens, sampler)


def choice(
    model,
    choices: List[str],
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial,
):
    regex_str = r"(" + r"|".join(choices) + r")"
    return regex(model, regex_str, max_tokens, sampler)


def json(
    model,
    schema_object: Union[str, object, Callable],
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial,
):
    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)

    return generator
