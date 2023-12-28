import json as pyjson
import warnings
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
    def __init__(self, fsm, model, sampler, device, max_tokens=None, stop_at=None):
        self.generate_token = token_generator(model, sampler)
        self.fsm = fsm
        self.tokenizer = model.tokenizer
        self.device = device
        self.max_tokens = max_tokens

        if isinstance(stop_at, str):
            stop_at = [stop_at]
        self.stop_sequences = stop_at

        if stop_at is not None:
            warnings.warn(
                "The use of the `stop_at` keyword when initiating a SequenceGenerator is deprecated, "
                "please use it when calling the genetator instead. "
                "The parameter will be removed in Outlines v0.1.0.",
                DeprecationWarning,
            )
        if max_tokens is not None:
            warnings.warn(
                "The use of the `max_tokens` keyword when initiating a SequenceGenerator is deprecated, "
                "please use it when calling the genetator instead. "
                "The parameter will be removed in Outlines v0.1.0.",
                DeprecationWarning,
            )

    def get_generated_token_ids(
        self,
        init_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prompts: List[str],
        last_state: GenerationState,
    ) -> List[torch.Tensor]:
        """Get the tokens generated so far.

        Parameters
        ----------
        init_state
            The initial state of the generation.
        prompts
            The prompts passed to the generator.
        last_state
            The current state of the generation

        Returns
        -------
        A tensor that contains the token ids that have been generated so far.

        """
        prompt_token_ids = init_state[0]
        prompt_lengths = [len(prompt_token_ids[i]) for i in range(len(prompts))]

        token_ids = [
            cur_token_ids[length:]
            for cur_token_ids, length in zip(last_state.token_ids, prompt_lengths)
        ]

        return token_ids

    def is_stop_sequence_found(
        self, generated_sequences: List[str], stop_sequences: List[str]
    ) -> bool:
        """Determine whether one of the stop sequences has been generated.

        Parameters
        ----------
        generated_sequences
            The list of sequences generated so far.
        stop_sequences
            The list that contains the sequence which stop the generation when
            found.

        Returns
        -------
        True if at least one of the stop sequences has been found in each generated
        sequence.

        """
        return all(
            [
                any([seq in generated for seq in stop_sequences])
                for generated in generated_sequences
            ]
        )

    def strip_stop_sequences(self, sequence: str, stop_sequences: List[str]) -> str:
        """Remove the stop sequences from the generated sequences.

        Parameters
        ----------
        sequence
            One of the generated sequences.
        stop_sequences
            The list that contains the sequence which stop the generation when
            found.

        """
        if stop_sequences:
            match_indexes = [sequence.find(seq) for seq in stop_sequences]
            if any([index != -1 for index in match_indexes]):
                # select the stop_sequence that is found first in the sequence
                min_match_index_value = min([i for i in match_indexes if i != -1])
                min_match_index_pos = match_indexes.index(min_match_index_value)
                sequence = sequence[
                    : match_indexes[min_match_index_pos]
                    + len(stop_sequences[min_match_index_pos])
                ]

        return sequence

    def format_sequence(self, sequence: str) -> str:
        """Translate the generated sequence to another type.

        This method is for instance overridden when generating JSON to either
        return a dictionnary or a Pydantic model.

        Parameters
        ----------
        sequence
            A generated sequences.

        Returns
        -------
        The formatted sequence.

        """
        return sequence

    def __call__(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
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
        max_tokens
            An integer representing maximum number of tokens that will be generated
            (per prompt)
        stop_at
            A string or list of strings at which the text generated will stop
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

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        stop_sequences = stop_at or self.stop_sequences
        max_tokens = max_tokens or self.max_tokens
        num_sequences = len(prompts)

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        init_state = init_generator_state(
            self.tokenizer, self.device, prompts, kv_cache
        )
        init_fsm_states = [FSMState(0) for _ in range(num_sequences)]

        states = sequence_generator(
            self.generate_token, self.fsm, init_state, init_fsm_states, rng
        )

        while True:
            try:
                last_state = next(states)
                if max_tokens or stop_sequences:
                    generated_token_ids = self.get_generated_token_ids(
                        init_state, prompts, last_state
                    )
                    if max_tokens and len(generated_token_ids[0]) >= max_tokens:
                        break
                    if stop_sequences and self.is_stop_sequence_found(
                        self.tokenizer.decode(generated_token_ids), stop_sequences
                    ):
                        break
            except StopIteration:
                break

        generated_token_ids = self.get_generated_token_ids(
            init_state, prompts, last_state
        )
        generated = self.tokenizer.decode(generated_token_ids)
        stripped = [
            self.strip_stop_sequences(sequence, stop_sequences)
            for sequence in generated
        ]
        try:
            formatted = [self.format_sequence(sequence) for sequence in stripped]
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
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
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
        max_tokens
            An integer representing maximum number of tokens that will be generated
            (per prompt)
        stop_at
            A string or list of strings at which the text generated will stop
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

        max_tokens = max_tokens or self.max_tokens

        if isinstance(stop_at, str):
            stop_at = [stop_at]
        stop_sequences = stop_at or self.stop_sequences

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
            is_stop_at_reached = [False for _ in range(num_sequences)]
            while True:
                if (max_tokens and num_generated >= max_tokens) or all(
                    is_stop_at_reached
                ):
                    return
                try:
                    sequence = next(states)
                    num_generated += 1
                except StopIteration:
                    return
                generated_token_ids = sequence.token_ids[:, -num_generated:]
                generated_sequences = self.tokenizer.decode(generated_token_ids)
                next_tokens = [
                    token[len(sequence) :] if not stop else ""
                    for token, sequence, stop in zip(
                        generated_sequences,
                        previously_generated_sequences,
                        is_stop_at_reached,
                    )
                ]
                previously_generated_sequences = generated_sequences
                if stop_sequences:
                    is_stop_at_reached = [
                        stop
                        or self.is_stop_sequence_found(
                            [generated_sequence], stop_sequences
                        )
                        for generated_sequence, stop in zip(
                            generated_sequences, is_stop_at_reached
                        )
                    ]
                yield next_tokens

        return token_generator()


def text(
    model,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    *,
    sampler: Sampler = multinomial,
):
    eos_token = model.tokenizer.eos_token_id
    fsm = StopAtTokenFSM(model.tokenizer, eos_token)

    device = model.device
    generator = SequenceGenerator(
        fsm, model, sampler, device, max_tokens=max_tokens, stop_at=stop_at
    )

    return generator


def regex(
    model,
    regex_str: str,
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial,
):
    fsm = RegexFSM(regex_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device, max_tokens=max_tokens)

    return generator


def cfg(
    model,
    cfg_str: str,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial,
):
    fsm = CFGFSM(cfg_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(
        fsm, model, sampler, device, max_tokens=max_tokens, stop_at=stop_at
    )

    return generator


def format(
    model,
    python_type,
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial,
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
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator
