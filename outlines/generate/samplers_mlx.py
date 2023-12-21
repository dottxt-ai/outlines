from typing import Protocol

import mlx.core as mx


class Sampler_mlx(Protocol):
    def __call__(
        self, logits: mx.array, samples: int, rng:mx.array, 
    ) -> mx.array:
        ...

def multinomial_mlx(
    logits: mx.array, samples: int,  rng:mx.array
) -> mx.array:
        """Multinomial sampling algorithm.

        Multinomial sampling consists in randomly sampling the next token assuming
        its distribution is a Categorical distribution parametrized by the
        next-token logits.

        Parameters
        ----------
        logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        samples
            The number of sequences to sample.
        rng
            A random number generator key. 
            Can be seeded using: key = mx.random.key(seed_value)
            (https://ml-explore.github.io/mlx/build/html/python/random.html)

        Returns
        -------
        The ids of the sampled tokens having shape ``(samples, n_seqs)``.

        """
        #key=rng,
        #print("rng key:", rng)
        #mx.random.seed(0)

        next_token_ids = mx.random.categorical(logits=logits, num_samples = samples)
        return next_token_ids
