import math

import pytest
import torch

from outlines.generate.generator import bias_logits, token_generator


def test_generator_error():
    def model(*_):
        raise IndexError

    def sampler():
        return None

    generator = token_generator(model, sampler, 1, None)
    next(generator)
    with pytest.raises(StopIteration):
        generator.send(((None, None, None), None))


@pytest.mark.parametrize(
    "logits,indices_to_mask,expected",
    [
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[]],
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[1]],
            torch.tensor([[1, -math.inf, 3, 4]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[1, 3]],
            torch.tensor([[1, -math.inf, 3, -math.inf]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
            [[0], [2]],
            torch.tensor([[-math.inf, 2, 3], [4, 5, -math.inf]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
            [[1], [0, 2]],
            torch.tensor(
                [[1, -math.inf, 3], [-math.inf, 5, -math.inf]], dtype=torch.float
            ),
        ),
    ],
)
def test_bias_logits(logits, indices_to_mask, expected):
    masked_logits = bias_logits(logits, indices_to_mask)
    assert torch.equal(masked_logits, expected)


def test_generator_1d():
    def model(*_):
        return torch.tensor([[0, 1, 2, 3]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits)

    # 1D, no bias
    generator = token_generator(model, sampler, 1, None)
    next(generator)
    result, _ = generator.send(((None, None, None), [[]]))
    assert result == 3

    # 1D, bias one
    generator = token_generator(model, sampler, 1, None)
    next(generator)
    result, _ = generator.send(((None, None, None), [[3]]))
    assert result == 2

    # 1D, bias two
    generator = token_generator(model, sampler, 1, None)
    next(generator)
    result, _ = generator.send(((None, None, None), [[2, 3]]))
    assert result == 1


def test_generator_2d():
    def model(*_):
        return torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, dim=1)

    # 2D, no bias
    generator = token_generator(model, sampler, 1, None)
    next(generator)
    result, _ = generator.send(((None, None, None), [[]]))
    assert torch.equal(result, torch.tensor([3, 3]))

    # 2D, bias one each
    generator = token_generator(model, sampler, 1, None)
    next(generator)
    result, _ = generator.send(((None, None, None), [[3], [3]]))
    assert torch.equal(result, torch.tensor([2, 2]))

    # 2D, bias one
    generator = token_generator(model, sampler, 1, None)
    next(generator)
    result, _ = generator.send(((None, None, None), [[3], []]))
    assert torch.equal(result, torch.tensor([2, 3]))

    # 2D, bias different number
    generator = token_generator(model, sampler, 1, None)
    next(generator)
    result, _ = generator.send(((None, None, None), [[3], [2, 3]]))
    assert torch.equal(result, torch.tensor([2, 1]))


@pytest.mark.xfail
def get_next_fsm_states():
    raise NotImplementedError


@pytest.mark.xfail
def get_next_instructions():
    raise NotImplementedError


@pytest.mark.xfail
def is_generation_finished():
    raise NotImplementedError


@pytest.mark.xfail
def update_token_ids():
    raise NotImplementedError


@pytest.mark.xfail
def update_attention_masks():
    raise NotImplementedError
