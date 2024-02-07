from typing import Protocol, Tuple

import torch


class Sampler(Protocol):
    particles: int

    def __call__(
        self,
        next_token_logits: torch.DoubleTensor,
        sequence_weights: torch.DoubleTensor,
        rng: torch.Generator,
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

    def __call__(
        self,
        next_token_logits: torch.DoubleTensor,
        sequence_weights: torch.DoubleTensor,
        _,
    ) -> torch.DoubleTensor:
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
        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        next_token_ids = torch.argmax(logprobs, dim=-1, keepdim=True)

        ancestors = torch.arange(next_token_logits.shape[0])
        weights = sequence_weights + torch.gather(logprobs, 1, next_token_ids).squeeze()

        return next_token_ids, ancestors, weights


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
        self,
        next_token_logits: torch.DoubleTensor,
        sequence_weights: torch.DoubleTensor,
        rng: torch.Generator,
    ) -> Tuple[torch.DoubleTensor, torch.DoubleTensor, torch.DoubleTensor]:
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

        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        ancestors = torch.arange(next_token_logits.shape[0])
        weights = sequence_weights + torch.gather(logprobs, 1, next_token_ids).squeeze()

        return next_token_ids, ancestors, weights


multinomial = MultinomialSampler
