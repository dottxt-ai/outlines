import math

import torch

from outlines.text.generate.sample import greedy, multinomial, vectorized_random_choice


def test_greedy():
    logits = torch.tensor([[1.0, 2.0, 5.0]])
    next_token_ids = greedy(logits, samples=1)
    assert next_token_ids.equal(torch.tensor([[2]]))

    next_token_ids = greedy(logits, samples=2)
    assert next_token_ids.equal(torch.tensor([[2], [1]]))

    logits = torch.tensor([[10.0, 0.0, 3.0], [-math.inf, 2.0, 5.0]])
    next_token_ids = greedy(logits, samples=1)
    assert next_token_ids.equal(torch.tensor([[0, 2]]))

    next_token_ids = greedy(logits, samples=2)
    assert next_token_ids.equal(torch.tensor([[0, 2], [2, 1]]))


def test_multinomial():
    rng = torch.Generator()
    rng.manual_seed(239)

    logits = torch.tensor([[1.0, 4.0, 5.0]])
    next_token_ids = multinomial(logits, 1, rng)
    assert next_token_ids.equal(torch.tensor([[2]]))

    next_token_ids = multinomial(logits, 2, rng)
    assert next_token_ids.equal(torch.tensor([[2], [1]]))

    logits = torch.tensor([[10.0, 0.0, 9.0], [-math.inf, 4.0, 5.0]])
    next_token_ids = multinomial(logits, 1, rng)
    assert next_token_ids.equal(torch.tensor([[0, 2]]))

    next_token_ids = multinomial(logits, 2, rng)
    assert next_token_ids.equal(torch.tensor([[0, 1], [2, 2]]))


def test_vectorized_random_choice():
    rng = torch.Generator()
    rng.manual_seed(0)

    probs = torch.tensor([[1, 0, 0, 0]])
    sample = vectorized_random_choice(rng, probs)
    assert sample.shape == (1, 1)
    assert torch.equal(sample, torch.zeros((1, 1)))

    probs = torch.tensor([[1, 0, 0, 0]])
    sample = vectorized_random_choice(rng, probs, samples=3)
    assert sample.shape == (3, 1)
    assert torch.equal(sample, torch.zeros((3, 1)))

    probs = torch.tile(torch.tensor([[1, 0, 0, 0]]), (2, 1))
    sample = vectorized_random_choice(rng, probs)
    assert sample.shape == (1, 2)
    assert torch.equal(sample, torch.zeros((1, 2)))

    probs = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
    sample = vectorized_random_choice(rng, probs, samples=3)
    assert sample.shape == (3, 2)
    assert torch.equal(sample, torch.tensor([[0, 1], [0, 1], [0, 1]]))

    probs = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]])
    sample = vectorized_random_choice(rng, probs, samples=3)
    assert sample.shape == (3, 2, 2)
    assert torch.equal(
        sample, torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    )
