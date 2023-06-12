from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


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
        self.max_tokens = max_tokens

    def is_finished(self, token_ids: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Determine whether we should stop the generation."""
        raise NotImplementedError(
            "`Sequence.is_finished` must be implemented by subclasses."
        )

    def postprocess_completions(self, completions: List[str]) -> List[str]:
        return completions

    def step(
        self,
        rng: Generator,
        token_ids: NDArray[np.int64],
        attention_mask: NDArray[np.int64],
        samples: int = 1,
    ) -> Tuple[NDArray[np.int64], NDArray[float]]:
        """Generate one or several tokens that complete the input sequence.

        The sampling step consists in using a model to generate next-token
        logits and then sample `samples`-many new tokens from a categorical
        distribution parametrized by these logits.

        Parameters
        ----------
        rng
            NumPy random number Generator instance
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

        # Sample `samples`-many new tokens
        next_token_ids = vectorized_random_choice(rng, probs, samples)

        # Add the missing `num_tokens` and `num_sample` dimensions
        next_token_ids = np.expand_dims(next_token_ids, -1)
        token_ids = np.expand_dims(token_ids, 0)

        # Expand the input `token_ids` array to be able to concatenate several
        # samples.
        if samples > 1:
            repetitions = (samples,) + (1,) * num_input_dims
            token_ids = np.tile(token_ids, repetitions)
            probs = np.tile(probs, repetitions)

        token_ids = np.concatenate([token_ids, next_token_ids], axis=-1)

        # Merge sample and batch dimensions by removing dimensions of length
        # 1. The shape of the resulting arrays is `new_batch_shape + (num_tokens,)`
        # and `new_batch_shape + (vocab_size,)` respectively.
        token_ids = np.atleast_2d(token_ids.squeeze())
        probs = np.atleast_2d(probs.squeeze())

        return token_ids, probs

    def expand_attention_mask(
        self, attention_mask: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Expand the attention mask after the last completion."""
        batch_shape = attention_mask.shape[:-1]
        attention_mask = np.concatenate(
            [attention_mask, np.broadcast_to([1], batch_shape + (1,))], axis=-1
        )
        return attention_mask

    def update_token_ids(
        self,
        is_finished: NDArray[np.bool_],
        token_ids: NDArray[np.int64],
        token_ids_unfinished: NDArray[np.int64],
    ) -> NDArray[np.int64]:
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
        new_token_ids = np.empty(batch_shape + (num_tokens + 1,), dtype=np.int64)

        token_ids_finished = token_ids[is_finished]
        batch_shape_finished = token_ids_finished.shape[:-1]
        token_ids_finished = np.concatenate(
            [
                token_ids_finished,
                np.broadcast_to(
                    [self.model.tokenizer.pad_token_id], batch_shape_finished + (1,)
                ),
            ],
            axis=-1,
        )

        new_token_ids[~is_finished] = token_ids_unfinished
        new_token_ids[is_finished] = token_ids_finished

        return new_token_ids

    def __call__(
        self,
        prompt: Union[str, List[str]],
        samples: int = 1,
        rng: Generator = np.random.default_rng(),
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
        token_ids, attention_mask = self.model.tokenizer.encode(prompt)
        num_prompt_tokens = token_ids.shape[-1]

        if samples > 1:
            token_ids, _ = self.step(rng, token_ids, attention_mask, samples)
            is_finished = self.is_finished(token_ids)

            num_batch_dims = token_ids.ndim - 1
            repetitions = (samples,) + (1,) * num_batch_dims
            attention_mask = np.tile(attention_mask, repetitions)
            attention_mask = self.expand_attention_mask(attention_mask)
        else:
            batch_shape = token_ids.shape[:-1]
            is_finished = np.zeros(batch_shape, dtype=np.bool_)

        while True:
            num_generated_tokens = token_ids.shape[-1] - num_prompt_tokens
            if np.all(is_finished) or num_generated_tokens == self.max_tokens:
                break

            token_ids_unfinished = token_ids[~is_finished]
            attention_mask_unfinished = attention_mask[~is_finished]
            token_ids_unfinished, _ = self.step(
                rng, token_ids_unfinished, attention_mask_unfinished
            )

            token_ids = self.update_token_ids(
                is_finished, token_ids, token_ids_unfinished
            )
            attention_mask = self.expand_attention_mask(attention_mask)
            is_finished[~is_finished] = self.is_finished(token_ids_unfinished).flatten()

        result = self.model.tokenizer.decode(token_ids)
        result = self.postprocess_completions(result)

        if len(result) == 1:
            return result[0]

        return result


vsearchsorted = np.vectorize(np.searchsorted, otypes=[int], signature="(n),()->()")


def vectorized_random_choice(
    rng: Generator,
    p: NDArray[np.float64],
    samples: int = 1,
):
    """Vectorized implementation of `np.random.choice`.

    `np.random.choice` does not support arrays of probability. This implements
    the equivalent of this function where the `p` argument can be a matrix.

    Note
    ----
    `searchsorted` might be more efficient here since the number of elements
    can be quite large.

    Parameters
    ----------
    rng
        NumPy random number Generator instance
    p
        An array of probability of shape `(num_probability_vectors, num_items)`
        that must sum to 1.
    samples
        The number of samples to take for each probability vector.

    Returns
    -------
    An array of shape `(num_samples, batch_size)`

    """

    cumsum = np.expand_dims(p.cumsum(axis=-1), 0)
    rand = rng.random((samples,) + p.shape[:-1])
    idx = vsearchsorted(cumsum, rand)

    return idx
