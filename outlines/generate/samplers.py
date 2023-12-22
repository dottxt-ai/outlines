from typing import Protocol

import torch


class Sampler(Protocol):
    def __call__(
        self, logits: torch.DoubleTensor, samples: int, rng: torch.Generator
    ) -> torch.DoubleTensor:
        ...


def greedy(logits: torch.DoubleTensor, samples: int, *_) -> torch.DoubleTensor:
    """Greedy Sampling algorithm.

    Greedy sampling consists in choosing the token with the largest
    likelihood at every step.

    Parameters
    ----------
    logits
        A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
        probability distribution of the next token over the vocabulary.
    samples
        The number of sequences to produce.  In this case, the top-`samples`
        logit values are returned.
    rng
        A random number generator.

    Returns
    -------
    The ids of the sampled tokens having shape ``(samples, n_seqs)``.

    """
    if samples == 1:
        next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        next_token_ids = torch.topk(
            logits, samples, dim=-1, largest=True, sorted=True
        ).indices.T

    return next_token_ids


def multinomial(
    logits: torch.DoubleTensor, samples: int, rng: torch.Generator
) -> torch.DoubleTensor:
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
        A random number generator.

    Returns
    -------
    The ids of the sampled tokens having shape ``(samples, n_seqs)``.

    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    next_token_ids = torch.multinomial(probs, num_samples=samples, generator=rng)
    return next_token_ids
