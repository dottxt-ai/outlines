import warnings
from typing import Iterator, List, Optional, Union

import torch

from outlines.fsm.fsm import FSMState
from outlines.generate.generator import sequence_generator, token_generator


class SequenceGenerator:
    def __init__(
        self,
        fsm,
        model,
        sampler,
        device,
        *,
        max_tokens=None,
        stop_at=None,
    ):
        self.generate_token = token_generator(model, sampler)
        self.fsm = fsm
        self.tokenizer = model.tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.num_particles = sampler.particles

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
        prompt_token_ids: torch.Tensor,
        prompts: List[str],
        token_ids: torch.Tensor,
        num_samples: int,
    ) -> List[torch.Tensor]:
        """Get the tokens generated so far.

        Parameters
        ----------
        init_state
            The initial state of the generation.
        prompts
            The prompts passed to the generator.
        token_ids
            The generated token ids.
        num_samples
            The number of samples taken for each sequence

        Returns
        -------
        A tensor that contains the token ids that have been generated so far.

        """
        prompt_lengths = [
            len(prompt_token_ids[i])
            for _ in range(num_samples)
            for i in range(len(prompts))
        ]

        token_ids = [
            cur_token_ids[length:]
            for cur_token_ids, length in zip(token_ids, prompt_lengths)
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
    ) -> Union[str, List[str], List[List[str]]]:
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

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        stop_sequences = stop_at or self.stop_sequences
        max_tokens = max_tokens or self.max_tokens
        num_samples = self.num_particles

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        prompt_token_ids, attention_masks = self.tokenizer.encode(prompts)
        prompt_token_ids = prompt_token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        # To draw multiple samples we repeat the prompt as many times
        # as there are samples. We copy the FSMs and initialize the
        # FSM states.
        num_samples = self.num_particles
        batch_size = len(prompts)

        prompt_token_ids = torch.repeat_interleave(prompt_token_ids, num_samples, dim=0)
        attention_masks = torch.repeat_interleave(attention_masks, num_samples, dim=0)
        fsm_states = [FSMState(0) for _ in range(batch_size * num_samples)]
        fsms = [self.fsm.copy() for _ in range(batch_size * num_samples)]

        states = sequence_generator(
            self.generate_token,
            fsms,
            prompt_token_ids,
            attention_masks,
            fsm_states,
            rng=rng,
        )

        while True:
            try:
                last_state = next(states)
                if max_tokens or stop_sequences:
                    token_ids = last_state.token_ids
                    generated_token_ids = self.get_generated_token_ids(
                        prompt_token_ids, prompts, token_ids, num_samples
                    )
                    if max_tokens and len(generated_token_ids[0]) >= max_tokens:
                        break
                    if stop_sequences and self.is_stop_sequence_found(
                        self.tokenizer.decode(generated_token_ids), stop_sequences
                    ):
                        break
            except StopIteration:
                break

        token_ids = last_state.token_ids
        generated_token_ids = self.get_generated_token_ids(
            prompt_token_ids, prompts, token_ids, num_samples
        )

        generated = self.tokenizer.decode(generated_token_ids)
        stripped = [
            self.strip_stop_sequences(sequence, stop_sequences)
            for sequence in generated
        ]
        formatted = [self.format_sequence(sequence) for sequence in stripped]

        # We reshape the output to (sample_size, batch_size)
        output = []
        step = len(prompts)
        for i in range(0, len(formatted), step):
            output.append(formatted[i : i + step])

        # We remove leading dimensions for the output
        if len(prompts) == 1 and num_samples == 1:
            return output[0][0]
        elif num_samples == 1:
            return output[0]
        else:
            return output

    def stream(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        rng: Optional[torch.Generator] = None,
        kv_cache: Optional[torch.tensor] = None,
    ) -> Iterator[Union[List[str], List[List[str]], str]]:
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

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        stop_sequences = stop_at or self.stop_sequences
        max_tokens = max_tokens or self.max_tokens
        num_samples = self.num_particles

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        prompt_token_ids, attention_masks = self.tokenizer.encode(prompts)
        prompt_token_ids = prompt_token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        # To draw multiple samples we repeat the prompt as many times
        # as there are samples. We copy the FSMs and initialize the
        # FSM states.
        num_samples = self.num_particles
        batch_size = len(prompts)

        prompt_token_ids = torch.repeat_interleave(prompt_token_ids, num_samples, dim=0)
        attention_masks = torch.repeat_interleave(attention_masks, num_samples, dim=0)
        fsm_states = [FSMState(0) for _ in range(batch_size * num_samples)]
        fsms = [self.fsm.copy() for _ in range(batch_size * num_samples)]

        states = sequence_generator(
            self.generate_token,
            fsms,
            prompt_token_ids,
            attention_masks,
            fsm_states,
            rng=rng,
        )

        def token_generator() -> Iterator[Union[List[str], str, List[List[str]]]]:
            previously_generated_sequences = [
                "" for _ in range(batch_size)
            ] * num_samples
            num_generated = 0
            is_stop_at_reached = [False for _ in range(batch_size)] * num_samples
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
                # We reshape the output to (sample_size, batch_size)
                output = []
                step = len(prompts)
                for i in range(0, len(next_tokens), step):
                    output.append(next_tokens[i : i + step])

                # We remove leading dimensions for the output
                if len(prompts) == 1 and num_samples == 1:
                    yield output[0][0]
                elif num_samples == 1:
                    yield output[0]
                else:
                    yield output

        return token_generator()
