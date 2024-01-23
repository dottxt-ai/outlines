from typing import Protocol

import torch


class Sampler(Protocol):
    particles: int

    def __call__(
        self, logits: torch.DoubleTensor, rng: torch.Generator
    ) -> torch.DoubleTensor:
        ...


class GreedySampler:
    """Greedy Sampling algorithm.

    Greedy sampling consists in choosing the token with the largest
    likelihood at every step.

    We don't allow more than one sample as this does not really make sense.

    """

    def __init__(self):
        self.particles = 1

    def __call__(self, logits: torch.DoubleTensor, *_) -> torch.DoubleTensor:
        """Call the greedy sampler.

        Parameters
        ----------
        logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        rng
            A random number generator.

        Returns
        -------
        The ids of the sampled tokens, of shape ``(n_seqs, 1)``

        """
        next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token_ids


greedy = GreedySampler


class MultinomialSampler:
    """Multinomial sampling algorithm.

    Multinomial sampling consists in randomly sampling the next token assuming
    its distribution is a Categorical distribution parametrized by the
    next-token logits.


    Attributes
    ----------
    particles
        The number of samples taken for each input sequence.

    """

    def __init__(self, samples: int = 1):
        self.particles = samples

    def __call__(
        self, logits: torch.DoubleTensor, rng: torch.Generator
    ) -> torch.DoubleTensor:
        """Call the multinomial sampler.

        Parameters
        ----------
        logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        rng
            A random number generator.

        Returns
        -------
        The ids of the sampled tokens having shape ``(samples, n_seqs)``.

        """
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        return next_token_ids


multinomial = MultinomialSampler
