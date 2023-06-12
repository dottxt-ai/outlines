from typing import Dict, List, Union

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from outlines.text.generate.sequence import Sequence, vectorized_random_choice


def test_vectorized_random_choice():
    rng = np.random.default_rng(0)

    probs = np.array([[1, 0, 0, 0]])
    sample = vectorized_random_choice(rng, probs)
    assert sample.shape == (1, 1)
    assert_array_equal(sample, np.zeros((1, 1)))

    probs = np.array([[1, 0, 0, 0]])
    sample = vectorized_random_choice(rng, probs, samples=3)
    assert sample.shape == (3, 1)
    assert_array_equal(sample, np.zeros((3, 1)))

    probs = np.tile(np.array([[1, 0, 0, 0]]), (2, 1))
    sample = vectorized_random_choice(rng, probs)
    assert sample.shape == (1, 2)
    assert_array_equal(sample, np.zeros((1, 2)))

    probs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    sample = vectorized_random_choice(rng, probs, samples=3)
    assert sample.shape == (3, 2)
    assert_array_equal(sample, [[0, 1], [0, 1], [0, 1]])

    probs = np.array([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]])
    sample = vectorized_random_choice(rng, probs, samples=3)
    assert sample.shape == (3, 2, 2)
    assert_array_equal(sample, [[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]]])


def test_sequence_error():
    with pytest.raises(NotImplementedError, match="must be implemented"):
        sequence = Sequence(None)
        sequence.is_finished(np.array([1]))


def ModelStep(logits):
    """Mock model to test `Sequence.step`"""

    logits = np.array([logits])

    def call(input_ids, *_):
        """Call the model.

        We first repeat the logits `num_sequences` times, and then
        reshape the resulting array to match the batch size.

        """
        import math

        batch_shape = input_ids.shape[:-1]
        vocab_shape = (logits.shape[-1],)
        shaped_logits = np.tile(logits, (math.prod(batch_shape), 1))
        return shaped_logits.reshape(batch_shape + vocab_shape)

    return call


def test_sequence_step():
    rng = np.random.default_rng(0)

    logits = np.array([0, 1, 0, 0])
    model = ModelStep(logits)

    sequence = Sequence(model)

    input_ids = np.array([[1, 2]])
    token_ids, probs = sequence.step(rng, input_ids, np.ones((1, 2)))
    assert_array_equal(token_ids, [[1, 2, 1]])
    assert probs.shape == (1, 4)


def test_sequence_step_batch():
    rng = np.random.default_rng(0)

    logits = np.array([0, 1, 0, 0])
    model = ModelStep(logits)

    sequence = Sequence(model)

    input_ids = np.array([[1, 2], [3, 4]])
    token_ids, probs = sequence.step(rng, input_ids, np.ones((2, 2)))
    assert_array_equal(token_ids, [[1, 2, 1], [3, 4, 1]])
    assert probs.shape == (2, 4)


def test_sequence_step_sample():
    rng = np.random.default_rng(0)

    logits = np.array([0, 1, 0, 0])
    model = ModelStep(logits)

    sequence = Sequence(model)
    input_ids = np.array([[1, 2]])
    token_ids, probs = sequence.step(rng, input_ids, np.ones((1, 2)), samples=3)
    assert_array_equal(token_ids, [[1, 2, 1], [1, 2, 1], [1, 2, 1]])
    assert probs.shape == (3, 4)


def test_sequence_sample_batch():
    rng = np.random.default_rng(0)

    logits = np.array([0, 1, 0, 0])
    model = ModelStep(logits)

    sequence = Sequence(model)
    input_ids = np.array([[1, 2, 1], [3, 4, 1]])
    token_ids, probs = sequence.step(rng, input_ids, np.ones((2, 3)), samples=3)
    assert_array_equal(
        token_ids,
        [
            [[1, 2, 1, 1], [3, 4, 1, 1]],
            [[1, 2, 1, 1], [3, 4, 1, 1]],
            [[1, 2, 1, 1], [3, 4, 1, 1]],
        ],
    )
    assert probs.shape == (3, 2, 4)


def test_sequence_step_loop():
    """Make sure that we can feed `step`'s output back as an input."""

    rng = np.random.default_rng(0)

    logits = np.array([0, 1, 0, 0])
    model = ModelStep(logits)

    sequence = Sequence(model)
    input_ids = np.array([[1, 2]])
    token_ids, _ = sequence.step(rng, input_ids, np.ones((1, 2)))
    token_ids, probs = sequence.step(rng, token_ids, np.ones((1, 3)))
    assert_array_equal(token_ids, [[1, 2, 1, 1]])
    assert probs.shape == (1, 4)

    input_ids = np.array([[1, 2], [3, 4]])
    token_ids, _ = sequence.step(rng, input_ids, np.ones((2, 2)))
    token_ids, probs = sequence.step(rng, token_ids, np.ones((2, 3)))
    assert_array_equal(token_ids, [[1, 2, 1, 1], [3, 4, 1, 1]])
    assert probs.shape == (2, 4)

    # The number of samples becomes the batch size at the next iteration.
    input_ids = np.array([[1, 2]])
    token_ids, _ = sequence.step(rng, input_ids, np.ones((1, 2)), samples=3)
    token_ids, probs = sequence.step(rng, token_ids, np.ones((3, 3)))
    assert_array_equal(token_ids, [[1, 2, 1, 1], [1, 2, 1, 1], [1, 2, 1, 1]])
    assert probs.shape == (3, 4)


def test_sequence_step_loop_general():
    rng = np.random.default_rng(0)

    logits = np.array([0, 1, 0, 0])
    model = ModelStep(logits)

    sequence = Sequence(model)
    input_ids = np.array([[1, 2, 1], [3, 4, 1]])
    token_ids, _ = sequence.step(rng, input_ids, np.ones((1, 3)), samples=3)
    result, _ = sequence.step(rng, token_ids, np.ones((3, 4)))
    assert result.shape == (3, 2, 5)
    assert_array_equal(
        result,
        [
            [[1, 2, 1, 1, 1], [3, 4, 1, 1, 1]],
            [[1, 2, 1, 1, 1], [3, 4, 1, 1, 1]],
            [[1, 2, 1, 1, 1], [3, 4, 1, 1, 1]],
        ],
    )


class TokenizerUpdateTokens:
    pad_token_id = -1


class ModelUpdateTokens:
    tokenizer = TokenizerUpdateTokens()


def test_update_token_ids_all_unfinished():
    sequence = Sequence(ModelUpdateTokens())

    previous_token_ids = np.array([[1, 1], [1, 1]])
    is_finished = np.array([False, False])
    token_ids_unfinished = np.array([[1, 1, 1], [1, 1, 1]])

    result = sequence.update_token_ids(
        is_finished, previous_token_ids, token_ids_unfinished
    )
    assert_array_equal(result, [[1, 1, 1], [1, 1, 1]])


def test_update_token_ids_some_unfinished():
    "Makes sure that the pad token is appended to finished sequences."
    sequence = Sequence(ModelUpdateTokens())

    previous_token_ids = np.array([[1, 1], [1, 1]])
    token_ids_unfinished = np.array([[1, 1, 1]])
    is_finished = np.array([True, False])
    result = sequence.update_token_ids(
        is_finished, previous_token_ids, token_ids_unfinished
    )
    assert_array_equal(result, [[1, 1, -1], [1, 1, 1]])


@pytest.mark.xfail
def test_update_token_ids_larger_dimensions():
    sequence = Sequence(ModelUpdateTokens())

    previous_token_ids = np.array([[1, 1], [1, 1]])
    is_finished = np.array([False, False])
    token_ids_unfinished = np.array([[1, 1, 1], [1, 1, 1]])
    result = sequence.update_token_ids(
        is_finished, previous_token_ids, token_ids_unfinished
    )
    assert_array_equal(result, [[1, 1, -1], [1, 1, 1]])


class MockModel:
    def __init__(self, tokenizer, logits):
        self.tokenizer = tokenizer
        self.logits = np.array(logits)
        self.iteration_idx = 0

    def __call__(self, input_ids, *_):
        import math

        batch_shape = input_ids.shape[:-1]
        vocab_shape = (self.logits.shape[-1],)
        shaped_logits = np.tile(
            self.logits[self.iteration_idx], (math.prod(batch_shape), 1)
        )
        self.iteration_idx += 1

        return shaped_logits.reshape(batch_shape + vocab_shape)


class MockTokenizer:
    def __init__(self, vocabulary: Dict[str, int]):
        self.vocabulary = vocabulary
        self.pad_token_id = -1

    def encode(self, prompts: Union[str, List[str]]):
        if isinstance(prompts, str):
            prompts = [prompts]

        token_ids = np.array([[self.vocabulary[prompt]] for prompt in prompts])
        attention_mask = np.ones_like(token_ids)

        return token_ids, attention_mask

    def decode(self, token_ids):
        return token_ids


def test_call_single_prompt():
    class FinishAfterTwo(Sequence):
        def __init__(self, model):
            super().__init__(model)
            self.iteration_idx = 0

        def is_finished(self, token_ids):
            """Finish generating the sequence after two iterations"""
            if self.iteration_idx == 0:
                self.iteration_idx += 1
                return np.array([False])
            else:
                return np.array([True])

    tokenizer = MockTokenizer({"Test": 0, "a": 1, "b": 2})
    model = MockModel(tokenizer, [[1, 0, 0], [0, 1, 0]])
    sequence = FinishAfterTwo(model)

    result = sequence("Test")
    assert_array_equal(result, [0, 0, 1])


def test_call_prompt_list():
    class Tokenizer:
        def __init__(self, vocabulary: Dict[str, int]):
            self.vocabulary = vocabulary
            self.pad_token_id = -1

        def __call__(self, prompts: List[str], **_):
            return {
                "input_ids": np.array([[self.vocabulary[prompt]] for prompt in prompts])
            }

        def batch_decode(self, token_ids):
            return token_ids

    class FinishAfterThree(Sequence):
        def __init__(self, model):
            super().__init__(model)
            self.iteration_idx = 0

        def is_finished(self, token_ids):
            """Finish generating the first sequence after two iteration and the
            second one after two iterations.

            """
            if self.iteration_idx == 0:
                self.iteration_idx += 1
                return np.array([False, False, False])
            elif self.iteration_idx == 1:
                self.iteration_idx += 1
                return np.array([True, False, True])
            else:
                return np.array([True])  # We only consider the unfinished sequences

    tokenizer = MockTokenizer(
        {"Test1": 0, "Test2": 1, "a": 2, "b": 3, "c": 4, "Test3": 5}
    )
    model = MockModel(
        tokenizer,
        [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]],
    )
    sequence = FinishAfterThree(model)

    result = sequence(["Test1", "Test2", "Test3"])
    assert_array_equal(result, [[0, 2, 3, -1], [1, 2, 3, 4], [5, 2, 3, -1]])


def test_call_single_prompt_samples():
    class FinishAfterTwo(Sequence):
        def __init__(self, model):
            super().__init__(model)
            self.iteration_idx = 0

        def is_finished(self, token_ids):
            if self.iteration_idx == 0:
                self.iteration_idx += 1
                return np.array([False, False, False])
            else:
                return np.array([True, True, True])

    tokenizer = MockTokenizer({"a": 0, "b": 1, "c": 2, "Test": 4})
    model = MockModel(tokenizer, [[1, 0, 0, 0], [0, 1, 0, 0]])
    sequence = FinishAfterTwo(model)
    result = sequence("Test", samples=3)
    assert_array_equal(result, [[4, 0, 1], [4, 0, 1], [4, 0, 1]])

    class FinishAfterOne(Sequence):
        def __init__(self, model):
            super().__init__(model)

        def is_finished(self, token_ids):
            return np.array([True, True, True])

    tokenizer = MockTokenizer({"a": 0, "b": 1, "c": 3, "Test": 4})
    model = MockModel(tokenizer, [[1, 0, 0, 0], [0, 1, 0, 0]])
    sequence = FinishAfterOne(model)
    result = sequence("Test", samples=3)
    assert_array_equal(result, [[4, 0], [4, 0], [4, 0]])


def test_call_prompt_list_samples():
    class FinishAfterThree(Sequence):
        def __init__(self, model):
            super().__init__(model)
            self.iteration_idx = 0

        def is_finished(self, token_ids):
            if self.iteration_idx == 0:
                self.iteration_idx += 1
                batch_shape = token_ids.shape[:-1]
                return np.zeros(batch_shape, dtype=np.bool_)
            elif self.iteration_idx == 1:
                self.iteration_idx += 1
                return np.array(
                    [[True, False, True], [True, False, True], [True, False, True]]
                )
            else:
                return np.array([True, True, True])

    tokenizer = MockTokenizer(
        {"a": 0, "b": 1, "c": 2, "Test1": 3, "Test2": 4, "Test3": 5}
    )
    model = MockModel(
        tokenizer, [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
    )
    sequence = FinishAfterThree(model)

    result = sequence(["Test1", "Test2", "Test3"], samples=3)
    assert_array_equal(
        result, np.tile([[3, 0, 1, -1], [4, 0, 1, 2], [5, 0, 1, -1]], (3, 1, 1))
    )
