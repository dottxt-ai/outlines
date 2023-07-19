import itertools
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import interegular
import torch

from outlines.text.parsing import find_partial_matches


class Sequence:
    """Represents a sequence generation method."""

    def __init__(self, model, max_tokens: Optional[int] = None):
        """Create a `Sequence` instance.

        Parameters
        ----------
        model
            The instance of the model used to generate next-token probabilities.
        max_tokens
            The maximum number of tokens that will be generated if no termination
            condition is met.

        """
        self.model = model
        self.device = model.device
        self.max_tokens = max_tokens
        self.pad_token_id = torch.tensor(
            model.tokenizer.pad_token_id, device=model.device
        )

    def create_proposal(
        self, generated_token_ids: torch.LongTensor, logits: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Create a new proposal from the next-token logits."""
        return logits

    def is_finished(self, token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Determine whether we should stop the generation."""
        raise NotImplementedError(
            "`Sequence.is_finished` must be implemented by subclasses."
        )

    def postprocess_completions(self, completions: List[str]) -> List[str]:
        return completions

    def step(
        self,
        rng: torch.Generator,
        num_prompt_tokens: int,
        token_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        samples: int = 1,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """Generate one or several tokens that complete the input sequence.

        The sampling step consists in using a model to generate next-token
        logits and then sample `samples`-many new tokens from a categorical
        distribution parametrized by these logits.

        Parameters
        ----------
        rng
            NumPy random number Generator instance
        num_prompt_tokens
            The number of tokens in the prompt.
        token_ids
            The token ids passed as an input to the model, of shape `batch_shape
            + (num_tokens,)`, where `num_tokens` is the sequences' length.
        samples
            The number of continuations to sample from the next-token probability
            distribution.

        Returns
        -------
        A tuple with an array of shape `new_batch_shape + (num_tokens+1,)`that
        contains the completed sequences (input token ids and generated token
        ids) and an array of shape `new_batch_shape + (vocab_size,)` that
        contains the next token probabilities.
        `new_batch_shape` is computed by removing dimensions of size one in
        `(samples,) + batch_shape`.

        """
        num_input_dims = token_ids.ndim
        probs = self.model(token_ids, attention_mask)
        probs = self.create_proposal(token_ids[:, num_prompt_tokens:], probs)
        probs = torch.nn.functional.softmax(probs, dim=-1)

        # Sample `samples`-many new tokens
        next_token_ids = vectorized_random_choice(rng, probs, samples)

        # Add the missing `num_tokens` and `num_sample` dimensions
        next_token_ids = torch.unsqueeze(next_token_ids, -1)
        token_ids = torch.unsqueeze(token_ids, 0)

        # Expand the input `token_ids` array to be able to concatenate several
        # samples.
        if samples > 1:
            repetitions = (samples,) + (1,) * num_input_dims
            token_ids = torch.tile(token_ids, repetitions)
            probs = torch.tile(probs, repetitions)

        token_ids = torch.concatenate([token_ids, next_token_ids], axis=-1)

        # Merge sample and batch dimensions by removing dimensions of length
        # 1. The shape of the resulting arrays is `new_batch_shape + (num_tokens,)`
        # and `new_batch_shape + (vocab_size,)` respectively.
        token_ids = torch.atleast_2d(token_ids.squeeze())
        probs = torch.atleast_2d(probs.squeeze())

        return token_ids, probs

    def expand_attention_mask(
        self, attention_mask: torch.LongTensor
    ) -> torch.LongTensor:
        """Expand the attention mask after the last completion."""
        batch_shape = attention_mask.shape[:-1]
        attention_mask = torch.concatenate(
            [
                attention_mask,
                torch.broadcast_to(
                    torch.tensor([1], device=self.device), batch_shape + (1,)
                ),
            ],
            axis=-1,
        )
        return attention_mask

    def update_token_ids(
        self,
        is_finished: torch.BoolTensor,
        token_ids: torch.LongTensor,
        token_ids_unfinished: torch.LongTensor,
    ) -> torch.LongTensor:
        """Update the array of token ids after the last completion.

        We only generate new tokens for the sequences that are not finished. We thus
        update the array with the new tokens, and append pad tokens to the finished
        sequences.

        Parameters
        ----------
        is_finished
            Boolean array that indicates which sequences are finished.
        token_ids
            Array that contains the sequences before the generation's last step.
        token_ids_unfinished
            Array that contains the sequences of the unfinished sequences
            after the generation's last step.

        Returns
        -------
        An array that contains the updated array that contains the sequences. We append
        pad tokens to the finished sequences.

        """
        batch_shape = token_ids.shape[:-1]
        num_tokens = token_ids.shape[-1]
        new_token_ids = torch.empty(
            batch_shape + (num_tokens + 1,), dtype=torch.int64, device=self.device
        )
        token_ids_finished = torch.concatenate(
            [
                token_ids[is_finished],
                torch.broadcast_to(
                    self.pad_token_id,
                    token_ids[is_finished].shape[:-1] + (1,),
                ),
            ],
            axis=-1,
        )

        new_token_ids[~is_finished] = token_ids_unfinished
        new_token_ids[is_finished] = token_ids_finished

        return new_token_ids

    def find_boundary_tokens(self, prompt: str) -> Dict[int, List[int]]:
        """Find a list of tokens that cross the prompt boundary."""

        vocabulary = {
            token_id: self.model.tokenizer.decode([token_id])[0]
            for token_id in range(len(self.model.tokenizer.vocabulary))
        }
        prompt_fsm = interegular.parse_pattern(prompt).to_fsm()

        prompt_token_ids, _ = self.model.tokenizer.encode(prompt)
        prompt_tokens = self.model.tokenizer.decode(prompt_token_ids[0])

        token_idx_in_prompt = [0] + list(
            itertools.accumulate([len(t) for t in prompt_tokens])
        )[:-1]

        boundary_tokens = defaultdict(list)
        for token_id, token in vocabulary.items():
            pmatches = find_partial_matches(prompt_fsm, token)
            for pmatch in pmatches:
                end_idx, states = pmatch
                if end_idx is not None and states[-1] == len(prompt):
                    if states[0] in token_idx_in_prompt:
                        boundary_tokens[token_idx_in_prompt.index(states[0])].append(
                            token_id
                        )

        return boundary_tokens

    def align_prompt_tokens(
        self, prompt: Union[str, List[str]], rng: torch.Generator
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Align the prompt with the vocabulary."""

        prompts = prompt
        if isinstance(prompts, str):
            prompts = [prompts]

        masks = []
        truncated_attention_masks = []
        truncated_token_idss = []
        attention_masks = []
        for prompt in prompts:
            boundary_tokens = self.find_boundary_tokens(prompt)

            token_ids, attention_mask = self.model.tokenizer.encode(prompt)
            token_ids = token_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            last_token = min(boundary_tokens.keys())
            truncated_token_ids = token_ids[:, :last_token]
            truncated_attention_mask = attention_mask[:, :last_token]

            allowed_tokens = boundary_tokens[last_token]
            mask = torch.full(
                (len(self.model.tokenizer.vocabulary),), -math.inf, device=self.device
            )
            mask[allowed_tokens] = 0

            masks.append(mask)
            truncated_attention_masks.append(truncated_attention_mask.squeeze())
            attention_masks.append(attention_mask.squeeze())
            truncated_token_idss.append(truncated_token_ids.squeeze())

        # Pad left and stack
        from torch.nn.utils.rnn import pad_sequence

        mask = torch.vstack(masks)
        truncated_attention_mask = pad_sequence(
            [t.flip(dims=[0]) for t in truncated_attention_masks],
            batch_first=True,
            padding_value=0,
        ).flip(dims=[1])
        attention_mask = pad_sequence(
            [a.flip(dims=[0]) for a in attention_masks],
            batch_first=True,
            padding_value=0,
        ).flip(dims=[1])
        truncated_token_ids = pad_sequence(
            [t.flip(dims=[0]) for t in truncated_token_idss],
            batch_first=True,
            padding_value=self.model.tokenizer.pad_token_id,
        ).flip(dims=[1])

        probs = self.model(truncated_token_ids, truncated_attention_mask)
        probs = probs + mask
        probs = torch.nn.functional.softmax(probs, dim=-1)
        next_token_ids = torch.multinomial(probs, num_samples=1)
        token_ids = torch.concatenate([truncated_token_ids, next_token_ids], axis=-1)

        return token_ids, attention_mask

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        samples: int = 1,
        rng: Optional[torch.Generator] = None,
    ) -> Union[str, List[str]]:
        """Generate a new sequence given a prompt.

        Parameters
        ----------
        prompt
            The input prompt.
        samples
            The number of samples to generate for each prompt.

        Returns
        -------
        The full sequence that contains the prompts and the generated string.

        """
        if rng is None:
            rng = torch.Generator(device=self.device)

        token_ids, attention_mask = self.align_prompt_tokens(prompt, rng)

        num_prompt_tokens = token_ids.shape[-1]

        if samples > 1:
            token_ids, _ = self.step(
                rng, num_prompt_tokens, token_ids, attention_mask, samples
            )
            is_finished = self.is_finished(token_ids)

            num_batch_dims = token_ids.ndim - 1
            repetitions = (samples,) + (1,) * num_batch_dims
            attention_mask = torch.tile(attention_mask, repetitions)
            attention_mask = self.expand_attention_mask(attention_mask)
        else:
            batch_shape = token_ids.shape[:-1]
            is_finished = torch.zeros(batch_shape, dtype=torch.bool, device=self.device)

        while True:
            num_generated_tokens = token_ids.shape[-1] - num_prompt_tokens
            if torch.all(is_finished) or num_generated_tokens == self.max_tokens:
                break

            updated_token_ids, _ = self.step(
                rng,
                num_prompt_tokens,
                token_ids[~is_finished],
                attention_mask[~is_finished],
            )
            token_ids = self.update_token_ids(is_finished, token_ids, updated_token_ids)
            attention_mask = self.expand_attention_mask(attention_mask)
            is_finished[~is_finished] = self.is_finished(
                updated_token_ids[:, num_prompt_tokens:]
            ).flatten()

        result = self.model.tokenizer.decode(token_ids)
        result = self.postprocess_completions(result)

        if len(result) == 1:
            return result[0]

        return result


def vectorized_random_choice(
    rng: torch.Generator,
    p: torch.FloatTensor,
    samples: int = 1,
):
    """Vectorized implementation of `np.random.choice`.

    `np.random.choice` does not support arrays of probability. This implements
    the equivalent of this function where the `p` argument can be a matrix.

    Note
    ----
    `torch.searchsorted` may be more efficient, but it is not implemented for
    every backend, for instance MPS.

    Parameters
    ----------
    rng
        Torch random number Generator instance
    p
        An array of probability of shape `(num_probability_vectors, num_items)`
        that must sum to 1.
    samples
        The number of samples to take for each probability vector.

    Returns
    -------
    An array of shape `(num_samples, batch_size)`

    """
    cumsum = torch.unsqueeze(p.cumsum(axis=-1), 0)
    rand = torch.rand(
        (samples,) + p.shape[:-1] + (1,), generator=rng, device=rng.device
    )
    idx = (cumsum < rand).sum(axis=-1)

    return idx
