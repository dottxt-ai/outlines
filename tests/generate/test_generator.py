import math
from typing import Generator

import pytest
import torch

from outlines.fsm.fsm import FSMState
from outlines.generate.api import SequenceGenerator
from outlines.generate.generator import (
    bias_logits,
    expand_attention_masks,
    get_allowed_tokens,
    get_next_fsm_states,
    is_generation_finished,
    sequence_generator,
    token_generator,
    update_token_ids,
)


def test_sequence_generator_class():
    class MockFSM:
        first_state = 0

        def next_state(self, state, next_token_ids):
            return 4

        def allowed_token_ids(self, *_):
            return [4]

        def is_final_state(self, _):
            return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            # Input: "test"
            return torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 1, 1, 1]])

        def decode(self, tokens):
            return ["testx"[i] for i in tokens]

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.float), None

    class sampler:
        def __init__(self):
            self.particles = 1

        def __call__(self, biased_logits, *_):
            return torch.argmax(biased_logits, keepdims=True)

    # Stream
    generator = SequenceGenerator(MockFSM(), MockModel(), sampler(), "cpu")
    assert generator.device == "cpu"
    assert isinstance(generator.tokenizer, MockTokenizer)
    assert isinstance(generator.fsm, MockFSM)
    assert callable(generator.generate_token)

    sequence = generator.stream("test")
    assert isinstance(sequence, Generator)

    next(sequence)

    with pytest.raises(StopIteration):
        next(sequence)

    # Call
    generator = SequenceGenerator(MockFSM(), MockModel(), sampler(), "cpu")
    result = generator("test")
    assert result == "x"


def test_sequence_generator_1d_single_iteration():
    class MockFSM:
        def next_state(self, state, next_token_ids):
            return 0

        def allowed_token_ids(self, _):
            return [0, 1, 2, 3]

        def is_final_state(self, _):
            return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            return torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 1, 1, 1]])

        def decode(self, x):
            return x

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, keepdims=True)

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[1, 1, 1, 1]]),
    )
    init_fsm_states = [0]
    generate = token_generator(MockModel(), sampler)
    sequence = sequence_generator(
        generate,
        [MockFSM()],
        token_ids,
        attention_mask,
        init_fsm_states,
        rng=torch.Generator(),
    )
    result = next(sequence)

    assert torch.equal(result.token_ids, torch.tensor([[0, 1, 2, 3, 3]]))
    assert torch.equal(result.logits, torch.tensor([[0, 1, 2, 3]]))

    with pytest.raises(StopIteration):
        next(sequence)


def test_sequence_generator_1d_several_iterations():
    class MockFSM:
        def next_state(self, state, next_token_ids):
            return FSMState(state + 1)

        def allowed_token_ids(self, _):
            return [0, 1, 2, 3]

        def is_final_state(self, state):
            if state < 2:
                return False
            else:
                return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            return torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 1, 1, 1]])

        def decode(self, x):
            return x

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, keepdims=True)

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[1, 1, 1, 1]]),
    )
    init_fsm_states = [0]
    generate = token_generator(MockModel(), sampler)
    sequence = sequence_generator(
        generate,
        [MockFSM()],
        token_ids,
        attention_mask,
        init_fsm_states,
        rng=torch.Generator(),
    )

    result = next(sequence)
    assert torch.equal(result.token_ids, torch.tensor([[0, 1, 2, 3, 3]]))
    assert torch.equal(result.logits, torch.tensor([[0, 1, 2, 3]]))

    result = next(sequence)
    assert torch.equal(result.token_ids, torch.tensor([[0, 1, 2, 3, 3, 3]]))
    assert torch.equal(result.logits, torch.tensor([[0, 1, 2, 3]]))

    with pytest.raises(StopIteration):
        next(sequence)


def test_sequence_generator_2d_single_iteration():
    class MockFSM:
        def next_state(self, state, next_token_ids):
            return 0

        def allowed_token_ids(self, _):
            return [0, 1, 2, 3]

        def is_final_state(self, _):
            return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 1]]
            )

        def decode(self, x):
            return x

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, keepdims=True, dim=-1)

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
    )
    init_fsm_states = [0, 0]
    fsms = [MockFSM(), MockFSM()]
    generate = token_generator(MockModel(), sampler)
    sequence = sequence_generator(
        generate,
        fsms,
        token_ids,
        attention_mask,
        init_fsm_states,
        rng=torch.Generator(),
    )

    result = next(sequence)
    assert torch.equal(
        result.token_ids, torch.tensor([[0, 1, 2, 3, 3], [4, 5, 6, 7, 2]])
    )
    assert torch.equal(
        result.logits, torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float)
    )

    with pytest.raises(StopIteration):
        next(sequence)


def test_sequence_generator_2d_several_iterations():
    class MockFSM:
        def next_state(self, state, next_token_ids):
            return FSMState(state + 1)

        def allowed_token_ids(self, _):
            return [0, 1, 2, 3]

        def is_final_state(self, state):
            if state < 2:
                return False
            else:
                return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 1]]
            )

        def decode(self, x):
            return x

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, keepdims=True, dim=-1)

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
    )
    init_fsm_states = [0, 0]
    fsms = [MockFSM(), MockFSM()]
    generate = token_generator(MockModel(), sampler)
    sequence = sequence_generator(
        generate,
        fsms,
        token_ids,
        attention_mask,
        init_fsm_states,
        rng=torch.Generator(),
    )

    result = next(sequence)
    assert torch.equal(
        result.token_ids, torch.tensor([[0, 1, 2, 3, 3], [4, 5, 6, 7, 2]])
    )
    assert torch.equal(
        result.logits, torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float)
    )

    result = next(sequence)
    assert torch.equal(
        result.token_ids, torch.tensor([[0, 1, 2, 3, 3, 3], [4, 5, 6, 7, 2, 2]])
    )
    assert torch.equal(
        result.logits, torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float)
    )

    with pytest.raises(StopIteration):
        next(sequence)


def test_generator_error():
    def model(*_):
        raise IndexError

    def sampler():
        return None

    generator = token_generator(model, sampler)
    with pytest.raises(IndexError, match="The input length"):
        generator(None, None, None, None, None)


@pytest.mark.parametrize(
    "logits_biases,expected_result,expected_biased_logits",
    [
        ([[0, 1, 2, 3]], [[3]], [[0, 1, 2, 3]]),
        ([[0, 1, 2]], [[2]], [[0, 1, 2, -math.inf]]),
        ([[0, 1]], [[1]], [[0, 1, -math.inf, -math.inf]]),
    ],
)
def test_generator_1d(logits_biases, expected_result, expected_biased_logits):
    class MockTokenizer:
        def decode(self, _):
            return "x"

    class MockModel:
        tokenizer = MockTokenizer()

        def __call__(self, *_):
            return torch.tensor([[0, 1, 2, 3]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, keepdims=True)

    generator = token_generator(MockModel(), sampler)
    result, _, _, biased_logits = generator(None, None, None, logits_biases, None)
    assert torch.equal(result, torch.tensor(expected_result))
    assert torch.equal(biased_logits, torch.tensor(expected_biased_logits))


@pytest.mark.parametrize(
    "logits_biases,expected_result,expected_biased_logits",
    [
        ([[0, 1, 2, 3], [0, 1, 2, 3]], [[3], [3]], [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (
            [[0, 1, 2], [0, 1, 2]],
            [[2], [2]],
            [[0, 1, 2, -math.inf], [4, 5, 6, -math.inf]],
        ),
        ([[0, 1, 2], [0, 1, 2, 3]], [[2], [3]], [[0, 1, 2, -math.inf], [4, 5, 6, 7]]),
        (
            [[0, 1], [0, 1, 2]],
            [[1], [2]],
            [[0, 1, -math.inf, -math.inf], [4, 5, 6, -math.inf]],
        ),
    ],
)
def test_generator_2d(logits_biases, expected_result, expected_biased_logits):
    class MockTokenizer:
        def decode(self, _):
            return "x"

    class MockModel:
        tokenizer = MockTokenizer()

        def __call__(self, *_):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, dim=1, keepdims=True)

    generator = token_generator(MockModel(), sampler)
    result, _, _, biased_logits = generator(None, None, None, logits_biases, None)
    assert torch.equal(result, torch.tensor(expected_result))
    assert torch.equal(biased_logits, torch.tensor(expected_biased_logits))


def test_get_next_fsm_states():
    class MockFSM:
        def next_state(self, state, next_token_ids):
            return 0

        def copy(self):
            return self

    result = get_next_fsm_states([MockFSM()], [0], torch.tensor([[0]]))
    assert result == [0]

    result = get_next_fsm_states(
        [MockFSM(), MockFSM()], [0, 0], torch.tensor([[0], [0]])
    )
    assert result == [0, 0]


def test_get_allowed_token_idss():
    class MockFSM:
        def allowed_token_ids(self, _):
            return [1, 2, 3, 4]

    result = get_allowed_tokens([MockFSM()], [0])
    assert result == [[1, 2, 3, 4]]

    result = get_allowed_tokens([MockFSM(), MockFSM()], [0, 1])
    assert result == [[1, 2, 3, 4], [1, 2, 3, 4]]


def test_is_generation_finished():
    class MockFSMFinished:
        def is_final_state(self, _):
            return True

    result = is_generation_finished([MockFSMFinished(), MockFSMFinished()], [1, 1])
    assert result is True

    class MockFSMNotFinished:
        def is_final_state(self, state):
            if state == 0:
                return False
            else:
                return True

    result = is_generation_finished(
        [MockFSMNotFinished(), MockFSMNotFinished()], [0, 1]
    )
    assert result is False


@pytest.mark.parametrize(
    "token_ids,next_token_ids,expected_result",
    [
        (torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[1, 2]])),
        (
            torch.tensor([[1], [1]]),
            torch.tensor([[2], [3]]),
            torch.tensor([[1, 2], [1, 3]]),
        ),
    ],
)
def test_update_token_ids(token_ids, next_token_ids, expected_result):
    result = update_token_ids(token_ids, next_token_ids)
    assert torch.equal(result, expected_result)


@pytest.mark.parametrize(
    "attention_masks,expected_result",
    [
        (
            torch.tensor([[1, 1]], dtype=torch.float),
            torch.tensor([[1, 1, 1]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 1], [1, 1]], dtype=torch.float),
            torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float),
        ),
    ],
)
def test_expand_attention_masks(attention_masks, expected_result):
    result = expand_attention_masks(attention_masks)
    assert torch.equal(result, expected_result)


@pytest.mark.parametrize(
    "logits,indices_to_mask,expected",
    [
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[0, 1, 2, 3]],
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[0, 2, 3]],
            torch.tensor([[1, -math.inf, 3, 4]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[0, 2]],
            torch.tensor([[1, -math.inf, 3, -math.inf]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
            [[1, 2], [0, 1]],
            torch.tensor([[-math.inf, 2, 3], [4, 5, -math.inf]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
            [[0, 2], [1]],
            torch.tensor(
                [[1, -math.inf, 3], [-math.inf, 5, -math.inf]], dtype=torch.float
            ),
        ),
    ],
)
def test_bias_logits(logits, indices_to_mask, expected):
    masked_logits = bias_logits(logits, indices_to_mask)
    assert torch.equal(masked_logits, expected)
