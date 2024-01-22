import math

import torch

from outlines.generate.samplers import (
    GreedySampler,
    MultinomialSampler,
    greedy,
    multinomial,
)


def test_aliases():
    assert greedy == GreedySampler
    assert multinomial == MultinomialSampler


def test_greedy():
    sampler = GreedySampler()
    logits = torch.tensor([[1.0, 2.0, 5.0]])
    next_token_ids = sampler(logits)
    assert next_token_ids.equal(torch.tensor([[2]]))

    sampler = GreedySampler(2)
    next_token_ids = sampler(logits)
    assert next_token_ids.equal(torch.tensor([[2], [1]]))

    sampler = GreedySampler()
    logits = torch.tensor([[10.0, 0.0, 3.0], [-math.inf, 2.0, 5.0]])
    next_token_ids = sampler(logits)
    assert next_token_ids.equal(torch.tensor([[0], [2]]))

    sampler = GreedySampler(2)
    next_token_ids = sampler(logits)
    assert next_token_ids.equal(torch.tensor([[0, 2], [2, 1]]))


def test_multinomial():
    rng = torch.Generator()
    rng.manual_seed(239)

    sampler = MultinomialSampler()
    logits = torch.tensor([[1.0, 4.0, 5.0]])
    next_token_ids = sampler(logits, rng)
    assert next_token_ids.equal(torch.tensor([[2]]))

    sampler = MultinomialSampler()
    logits = torch.tensor([[10.0, 0.0, 9.0], [-math.inf, 4.0, 5.0]])
    next_token_ids = sampler(logits, rng)
    assert next_token_ids.equal(torch.tensor([[0], [2]]))
