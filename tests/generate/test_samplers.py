import math

import torch

from outlines.generate.samplers import greedy, multinomial


def test_greedy():
    logits = torch.tensor([[1.0, 2.0, 5.0]])
    next_token_ids = greedy(logits, samples=1)
    assert next_token_ids.equal(torch.tensor([[2]]))

    next_token_ids = greedy(logits, samples=2)
    assert next_token_ids.equal(torch.tensor([[2], [1]]))

    logits = torch.tensor([[10.0, 0.0, 3.0], [-math.inf, 2.0, 5.0]])
    next_token_ids = greedy(logits, samples=1)
    assert next_token_ids.equal(torch.tensor([[0], [2]]))

    next_token_ids = greedy(logits, samples=2)
    assert next_token_ids.equal(torch.tensor([[0, 2], [2, 1]]))


def test_multinomial():
    rng = torch.Generator()
    rng.manual_seed(239)

    logits = torch.tensor([[1.0, 4.0, 5.0]])
    next_token_ids = multinomial(logits, 1, rng)
    assert next_token_ids.equal(torch.tensor([[2]]))

    next_token_ids = multinomial(logits, 2, rng)
    assert next_token_ids.equal(torch.tensor([[2, 1]]))

    logits = torch.tensor([[10.0, 0.0, 9.0], [-math.inf, 4.0, 5.0]])
    next_token_ids = multinomial(logits, 1, rng)
    assert next_token_ids.equal(torch.tensor([[0], [1]]))

    next_token_ids = multinomial(logits, 2, rng)
    assert next_token_ids.equal(torch.tensor([[2, 0], [2, 1]]))
