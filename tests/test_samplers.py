import math

import pytest
import torch

from outlines.samplers import (
    BeamSearchSampler,
    GreedySampler,
    MultinomialSampler,
    beam_search,
    greedy,
    keep_top_k_logits,
    keep_top_p_logits,
    multinomial,
    rescale_logits,
)


def compute_logprobs(logits):
    return torch.nn.functional.log_softmax(logits, dim=-1)


def test_aliases():
    assert greedy == GreedySampler
    assert multinomial == MultinomialSampler
    assert beam_search == BeamSearchSampler


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


def test_multinomial_init():
    sampler = MultinomialSampler()
    assert sampler.logits_processors == []

    sampler = MultinomialSampler(3)
    assert sampler.logits_processors == []

    sampler = MultinomialSampler(top_k=1)
    assert len(sampler.logits_processors) == 1

    sampler = MultinomialSampler(top_p=0.95)
    assert len(sampler.logits_processors) == 1

    sampler = MultinomialSampler(top_k=1, top_p=0.95)
    assert len(sampler.logits_processors) == 1

    sampler = MultinomialSampler(temperature=1.0)
    assert len(sampler.logits_processors) == 1

    sampler = MultinomialSampler(top_k=1, temperature=1.0)
    assert len(sampler.logits_processors) == 2

    sampler = MultinomialSampler(top_p=0.95, temperature=1.0)
    assert len(sampler.logits_processors) == 2


def test_top_k():
    with pytest.raises(ValueError, match="`k` must be a strictly"):
        keep_top_k_logits(-1)

    with pytest.raises(ValueError, match="`k` must be a strictly"):
        keep_top_k_logits(0.1)

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    logits_processor = keep_top_k_logits(1)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[-math.inf, -math.inf, -math.inf, 4.0]]))

    logits_processor = keep_top_k_logits(10)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    logits_processor = keep_top_k_logits(2)
    result = logits_processor(logits)
    assert result.equal(
        torch.tensor(
            [[-math.inf, -math.inf, 3.0, 4.0], [-math.inf, -math.inf, 7.0, 8.0]]
        )
    )


def test_top_p():
    with pytest.raises(ValueError, match="`p` must be a floating point"):
        keep_top_p_logits(-0.1)

    with pytest.raises(ValueError, match="`p` must be a floating point"):
        keep_top_p_logits(0.0)

    with pytest.raises(ValueError, match="`p` must be a floating point"):
        keep_top_p_logits(1.1)

    logits = torch.tensor([[1.0, 1.01, 1.02, 4.0]])

    logits_processor = keep_top_p_logits(0.1)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[-math.inf, -math.inf, -math.inf, 4.0]]))

    logits_processor = keep_top_p_logits(0.95)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[-math.inf, 1.01, 1.02, 4.0]]))

    logits_processor = keep_top_p_logits(1.0)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[1.0, 1.01, 1.02, 4.0]]))

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    logits_processor = keep_top_p_logits(0.1)
    result = logits_processor(logits)
    assert result.equal(
        torch.tensor(
            [
                [-math.inf, -math.inf, -math.inf, 4.0],
                [-math.inf, -math.inf, -math.inf, 8.0],
            ]
        )
    )

    logits_processor = keep_top_p_logits(0.95)
    result = logits_processor(logits)
    assert result.equal(
        torch.tensor(
            [
                [-math.inf, 2.0, 3.0, 4.0],
                [-math.inf, 6.0, 7.0, 8.0],
            ]
        )
    )

    logits_processor = keep_top_p_logits(1.0)
    result = logits_processor(logits)
    assert result.equal(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
    )


def test_rescale():
    with pytest.raises(ValueError, match="`temperature` must"):
        rescale_logits(1)

    with pytest.raises(ValueError, match="`temperature` must"):
        rescale_logits(-0.1)

    with pytest.raises(ValueError, match="Please use the greedy sampler"):
        rescale_logits(0.0)


def test_beam_search():
    # Two beams, single sequence
    sampler = BeamSearchSampler(2)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    init_weights = torch.tensor([0, 1.0])
    next_token_ids, ancestors, weights = sampler(logits, init_weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[0], [1]]))
    assert ancestors.equal(torch.tensor([1, 0]))
    assert weights.equal(
        torch.tensor([init_weights[1] + logprobs[1][0], logprobs[0][1]])
    )

    # Make sure that initial samples are different
    sampler = BeamSearchSampler(2)
    logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    init_weights = torch.tensor([0, 0])
    next_token_ids, ancestors, weights = sampler(logits, init_weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[1], [0]]))
    assert ancestors.equal(torch.tensor([0, 0]))
    assert weights.equal(torch.tensor([logprobs[0][1], logprobs[0][0]]))

    # One beam, batch of two sequences. Reduces to Greedy Search.
    sampler = BeamSearchSampler(1)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    weights = torch.tensor([0, 0])
    next_token_ids, ancestors, weights = sampler(logits, weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[1], [0]]))
    assert ancestors.equal(torch.tensor([0, 1]))
    assert weights.equal(torch.tensor([logprobs[0][1], logprobs[1][0]]))

    # Two beams, batch of two sequences
    sampler = BeamSearchSampler(2)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 2.0], [0.0, 1.0]])
    init_weights = torch.tensor([0, 0, 2.0, 0])
    next_token_ids, ancestors, weights = sampler(logits, init_weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[0], [1], [0], [1]]))
    assert ancestors.equal(torch.tensor([1, 0, 2, 2]))
    assert weights.equal(
        torch.tensor(
            [
                logprobs[1][0],
                logprobs[0][1],
                init_weights[2] + logprobs[2][0],
                init_weights[2] + logprobs[2][1],
            ]
        )
    )
