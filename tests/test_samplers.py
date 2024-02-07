import math

import torch

from outlines.samplers import GreedySampler, MultinomialSampler, greedy, multinomial


def compute_logprobs(logits):
    return torch.nn.functional.log_softmax(logits, dim=-1)


def test_aliases():
    assert greedy == GreedySampler
    assert multinomial == MultinomialSampler


def test_greedy():
    sampler = GreedySampler()
    logits = torch.tensor([[1.0, 2.0, 5.0]])
    weights = torch.tensor([0])
    next_token_ids, ancestors, weights = sampler(logits, weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[2]]))
    assert ancestors.equal(torch.tensor([0]))
    assert weights.equal(logprobs[..., 2])

    sampler = GreedySampler()
    logits = torch.tensor([[10.0, 0.0, 3.0], [-math.inf, 2.0, 5.0]])
    weights = torch.tensor([0, 0])
    next_token_ids, ancestors, weights = sampler(logits, weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[0], [2]]))
    assert ancestors.equal(torch.tensor([0, 1]))
    assert weights.equal(torch.tensor([logprobs[0, 0], logprobs[1, 2]]))


def test_multinomial():
    rng = torch.Generator()
    rng.manual_seed(239)

    sampler = MultinomialSampler()
    logits = torch.tensor([[1.0, 4.0, 5.0]])
    weights = torch.tensor([0])
    next_token_ids, ancestors, weights = sampler(logits, weights, rng)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[2]]))
    assert ancestors.equal(torch.tensor([0]))
    assert weights.equal(torch.tensor([logprobs[:, 2]]))

    sampler = MultinomialSampler()
    logits = torch.tensor([[10.0, 0.0, 9.0], [-math.inf, 4.0, 5.0]])
    weights = torch.tensor([0, 0])
    next_token_ids, ancestors, weights = sampler(logits, weights, rng)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[0], [2]]))
    assert ancestors.equal(torch.tensor([0, 1]))
    assert weights.equal(torch.tensor([logprobs[0, 0], logprobs[1, 2]]))
