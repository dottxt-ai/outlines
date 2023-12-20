import json as pyjson
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel

from outlines.fsm.fsm import CFGFSM, FSMState, RegexFSM, StopAtTokenFSM
from outlines.fsm.json_schema import build_regex_from_object, get_schema_from_signature
from outlines.fsm.types import python_types_to_regex
from outlines.generate.generator import (
    GenerationState,
    init_generator_state,
    sequence_generator,
    token_generator,
)
from outlines.generate.samplers import Sampler, multinomial


class SequenceGenerator:
    def __init__(self, fsm, model, sampler, device, stop=None, max_tokens=None):
        self.generate_token = token_generator(model, sampler)
        self.fsm = fsm
        self.tokenizer = model.tokenizer
        self.device = device
        self.max_tokens = max_tokens
        if isinstance(stop, str):
            stop = [stop]
        self.stop_sequences = stop

    def get_generated_token_ids(
        self,
        init_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prompts: List[str],
        last_state: GenerationState,
    ) -> List[torch.Tensor]:
        """Give the tokens generated (so the current sequences without the initial user prompts)"""
        # Get the number of tokens in the prompts
        prompt_token_ids = init_state[0]
        prompt_lengths = [len(prompt_token_ids[i]) for i in range(len(prompts))]
        # Remove the prompts from the generated sequences
        token_ids = [
            cur_token_ids[length:]
            for cur_token_ids, length in zip(last_state.token_ids, prompt_lengths)
        ]
        return token_ids

    def is_stop_sequence_reached(self, token_ids: List[torch.Tensor]) -> bool:
        """True if at least one of the stop sequences is found in each generated sequence"""
        return all(
            [
                any([seq in generated for seq in self.stop_sequences])
                for generated in self.tokenizer.decode(token_ids)
            ]
        )

    def format_sequence(self, sequence: str) -> str:
        """Format the text sequence generated before returning it to the user"""
        if self.stop_sequences:
            match_indexes = [sequence.find(seq) for seq in self.stop_sequences]
            if any([index != -1 for index in match_indexes]):
                # select the stop_sequence that is found first in the sequence
                min_match_index_value = min([i for i in match_indexes if i != -1])
                min_match_index_pos = match_indexes.index(min_match_index_value)
                return sequence[
                    : match_indexes[min_match_index_pos]
                    + len(self.stop_sequences[min_match_index_pos])
                ]
            return sequence
        return sequence

    def __call__(
        self,
        prompts: Union[str, List[str]],
        rng: Optional[torch.Generator] = None,
        kv_cache: Optional[torch.tensor] = None,
    ) -> Union[str, List[str]]:
        """Generate the full text sequence.

        Since `SequenceGenerator.stream` calls the tokenizer at every step this
        method loops over the generator returned by `sequence_generator` itself
        so the tokenizer is called only once after all token ids have been
        generated.

        Parameters
        ----------
        prompts
            A string or list of strings that are passed to the model before
            generating the first token.
        kv_cache
            A tensor containing the past key-value cache. It can be for instance
            used when we are interleaving prompting and model calls. Defaults to
            `None`.
        rng
            The random number generator. Defaults to a non-seeded `torch.Generator`
            instance.

        Returns
        -------
        A string or list of strings that contain the generated text.

        """

        self.fsm.reset()

        if isinstance(prompts, str):
            prompts = [prompts]

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        init_state = init_generator_state(
            self.tokenizer, self.device, prompts, kv_cache
        )
        num_sequences = len(prompts)
        init_fsm_states = [FSMState(0) for _ in range(num_sequences)]

        states = sequence_generator(
            self.generate_token, self.fsm, init_state, init_fsm_states, rng
        )

        while True:
            try:
                last_state = next(states)
                if self.max_tokens or self.stop_sequences:
                    token_ids = self.get_generated_token_ids(
                        init_state, prompts, last_state
                    )
                    if self.max_tokens and len(token_ids[0]) >= self.max_tokens:
                        break
                    if self.stop_sequences and self.is_stop_sequence_reached(token_ids):
                        break
            except StopIteration:
                break

        token_ids = self.get_generated_token_ids(init_state, prompts, last_state)
        generated = self.tokenizer.decode(token_ids)

        try:
            formatted = [self.format_sequence(sequence) for sequence in generated]
        except pyjson.decoder.JSONDecodeError:
            raise TypeError(
                "Could not format the output of the model into a dictionary or a Pydantic model."
                + " The model has likely exceeded its context length. Please try again using `constr` (for Pydantic)"
                + " and `maxLength` (for JSON Schema) to limit the length of the string fields. If this exception"
                + " is raised nevertheless please open an issue: https://github.com/outlines-dev/outlines/issues"
            )

        return formatted if len(formatted) > 1 else formatted[0]

    def stream(
        self,
        prompts: Union[str, List[str]],
        rng: Optional[torch.Generator] = None,
        kv_cache: Optional[torch.tensor] = None,
    ) -> Iterator[Union[List[str], str]]:
        """Generate the text sequence one token at a time.

        Since `Tokenizer.decode` strips the whitespaces from the tokens we have no
        choice but to decode the generated token ids at each step and compare the
        current decoded strings to the previously decoded strings.

        Parameters
        ----------
        prompts
            A string or list of strings that are passed to the model before
            generating the first token.
        kv_cache
            A tensor containing the past key-value cache. It can be for instance
            used when we are interleaving prompting and model calls. Defaults to
            `None`.
        rng
            The random number generator. Defaults to a non-seeded `torch.Generator`
            instance.

        Returns
        -------
        A string or list of strings that contain the generated text.

        """

        self.fsm.reset()

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        init_state = init_generator_state(
            self.tokenizer, self.device, prompts, kv_cache
        )

        token_ids = init_state[1]
        num_sequences = token_ids.shape[0]

        init_fsm_states = [FSMState(0) for _ in range(num_sequences)]

        states = sequence_generator(
            self.generate_token, self.fsm, init_state, init_fsm_states, rng
        )

        def token_generator() -> Iterator[Union[List[str], str]]:
            previously_generated_sequences = ["" for _ in range(num_sequences)]
            num_generated = 0
            while True:
                try:
                    sequence = next(states)
                    num_generated += 1
                except StopIteration:
                    return

                generated_token_ids = sequence.token_ids[:, -num_generated:]
                generated_sequences = self.tokenizer.decode(generated_token_ids)
                next_tokens = [
                    token[len(sequence) :]
                    for token, sequence in zip(
                        generated_sequences, previously_generated_sequences
                    )
                ]
                previously_generated_sequences = generated_sequences

                yield next_tokens

        return token_generator()


def text(
    model,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    *,
    sampler: Sampler = multinomial,
):
    eos_token = model.tokenizer.eos_token_id
    fsm = StopAtTokenFSM(model.tokenizer, eos_token)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device, stop, max_tokens)

    return generator


def regex(
    model,
    regex_str: str,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial,
):
    fsm = RegexFSM(regex_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device, stop, max_tokens)

    return generator


def cfg(
    model,
    cfg_str: str,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial,
):
    fsm = CFGFSM(cfg_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device, stop, max_tokens)

    return generator


def format(
    model,
    python_type,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial,
):
    regex_str = python_types_to_regex(python_type)
    return regex(model, regex_str, max_tokens, stop, sampler)


def choice(
    model,
    choices: List[str],
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial,
):
    regex_str = r"(" + r"|".join(choices) + r")"
    return regex(model, regex_str, max_tokens, stop, sampler)


def json(
    model,
    schema_object: Union[str, object, Callable],
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial,
):
    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, stop, sampler)
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, stop, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, stop, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator
