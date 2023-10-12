import math
from typing import Dict, List, Union

import numpy as np
import pytest
import torch

from outlines.models.tokenizer import Tokenizer
from outlines.text.generate.sequence import Sequence


class MockModel:
    def __init__(self, tokenizer, logits):
        self.tokenizer = tokenizer
        self.logits = logits
        self.iteration_idx = 0
        self.device = "cpu"

    def forward(self, input_ids, *_):
        import math

        batch_shape = input_ids.shape[:-1]
        vocab_shape = (self.logits.shape[-1],)
        shaped_logits = torch.tile(
            self.logits[self.iteration_idx], (math.prod(batch_shape), 1)
        )
        self.iteration_idx += 1

        return shaped_logits.reshape(batch_shape + vocab_shape), None

    def __call__(self, input_ids, *_):
        return self.forward(input_ids)[0]


class MockTokenizer(Tokenizer):
    def __init__(self, vocabulary: Dict[str, int]):
        self.vocabulary = vocabulary
        self.id_to_str = {v: k for k, v in vocabulary.items()} if vocabulary else {}
        self.pad_token_id = -1
        self.id_to_str[self.pad_token_id] = "<pad>"

    def encode(self, prompts: Union[str, List[str]]):
        if isinstance(prompts, str):
            prompts = [prompts]

        token_ids = torch.tensor([[self.vocabulary[prompt]] for prompt in prompts])
        attention_mask = torch.ones_like(token_ids)

        return token_ids, attention_mask

    def decode(self, token_ids):
        ndims = np.ndim(token_ids)

        assert 0 < ndims <= 2

        if ndims == 1:
            token_ids = [token_ids]

        res = ["".join(self.id_to_str[int(idx)] for idx in seq) for seq in token_ids]

        return res if ndims > 1 else res[0]

    def convert_token_to_string(self, token: str) -> str:
        return token

    def __hash__(self):
        return id(self)


def test_sequence_error():
    with pytest.raises(NotImplementedError, match="must be implemented"):
        sequence = Sequence(MockModel(MockTokenizer(None), None))
        sequence.is_finished(torch.tensor([1]))


class ModelStep:
    """Mock model to test `Sequence.step`"""

    def __init__(self, tokenizer, logits):
        self.device = "cpu"
        self.logits = logits
        self.tokenizer = tokenizer

    def forward(self, input_ids, *_):
        """Call the model.

        We first repeat the logits `num_sequences` times, and then
        reshape the resulting array to match the batch size.

        """
        import math

        batch_shape = input_ids.shape[:-1]
        vocab_shape = (self.logits.shape[-1],)
        shaped_logits = torch.tile(self.logits, (math.prod(batch_shape), 1))
        return shaped_logits.reshape(batch_shape + vocab_shape), None

    def __call__(self, input_ids, *_):
        return self.forward(input_ids)[0]


def test_sequence_step():
    rng = torch.Generator()
    rng.manual_seed(0)

    logits = torch.tensor([-math.inf, 1, -math.inf, -math.inf], dtype=torch.double)
    model = ModelStep(MockTokenizer(None), logits)

    sequence = Sequence(model)

    input_ids = torch.tensor([[1, 2]])
    token_ids, probs, _ = sequence.step(rng, 2, input_ids, torch.ones((1, 2)))
    assert torch.equal(token_ids, torch.tensor([[[1]]]))
    assert probs.shape == (1, 1, 4)


def test_sequence_step_batch():
    rng = torch.Generator()
    rng.manual_seed(0)

    logits = torch.tensor([-math.inf, 0.5, 0.5, -math.inf], dtype=torch.double)
    model = ModelStep(MockTokenizer(None), logits)

    sequence = Sequence(model)

    input_ids = torch.tensor([[1, 2], [3, 4]])
    token_ids, probs, _ = sequence.step(rng, 2, input_ids, torch.ones((2, 2)))
    assert torch.equal(token_ids, torch.tensor([[[1], [2]]]))
    assert probs.shape == (1, 2, 4)


def test_sequence_step_sample():
    rng = torch.Generator()
    rng.manual_seed(0)

    logits = torch.tensor([-math.inf, 0.5, 0.5, -math.inf], dtype=torch.double)
    model = ModelStep(MockTokenizer(None), logits)

    sequence = Sequence(model)
    input_ids = torch.tensor([[1, 2]])
    token_ids, probs, _ = sequence.step(
        rng, 2, input_ids, torch.ones((1, 2)), samples=3
    )
    assert torch.equal(token_ids, torch.tensor([[[1]], [[2]], [[1]]]))
    assert probs.shape == (3, 1, 4)


def test_sequence_step_sample_batch():
    rng = torch.Generator()
    rng.manual_seed(0)

    logits = torch.tensor([-math.inf, 0.5, 0.5, -math.inf], dtype=torch.double)
    model = ModelStep(MockTokenizer(None), logits)

    sequence = Sequence(model)
    input_ids = torch.tensor([[1, 2, 1], [3, 4, 1]])
    token_ids, probs, _ = sequence.step(
        rng, 3, input_ids, torch.ones((2, 3)), samples=3
    )
    assert torch.equal(
        token_ids,
        torch.tensor(
            [
                [[1], [2]],
                [[1], [1]],
                [[1], [2]],
            ]
        ),
    )
    assert probs.shape == (3, 2, 4)


def test_sequence_step_loop():
    """Make sure that we can feed `step`'s output back as an input."""
    rng = torch.Generator()
    rng.manual_seed(0)

    logits = torch.tensor([-math.inf, 0.5, 0.5, -math.inf], dtype=torch.double)
    model = ModelStep(MockTokenizer(None), logits)

    sequence = Sequence(model)
    input_ids = torch.tensor([[1, 2]])
    token_ids, *_ = sequence.step(rng, 2, input_ids, torch.ones((1, 2)))
    token_ids, probs, _ = sequence.step(
        rng, 2, token_ids.squeeze(0), torch.ones((1, 3))
    )
    assert torch.equal(token_ids, torch.tensor([[[2]]]))
    assert probs.shape == (1, 1, 4)

    input_ids = torch.tensor([[1, 2], [3, 4]])
    token_ids, *_ = sequence.step(rng, 2, input_ids, torch.ones((2, 2)))
    token_ids, probs, _ = sequence.step(
        rng, 2, token_ids.squeeze(0), torch.ones((2, 3))
    )
    assert torch.equal(token_ids, torch.tensor([[[1], [2]]]))
    assert probs.shape == (1, 2, 4)

    # The number of samples becomes the batch size at the next iteration.
    input_ids = torch.tensor([[1, 2]])
    token_ids, *_ = sequence.step(rng, 2, input_ids, torch.ones((1, 2)), samples=3)
    token_ids, probs, _ = sequence.step(
        rng, 2, token_ids.squeeze(1), torch.ones((3, 3))
    )
    assert torch.equal(token_ids, torch.tensor([[[2], [1], [1]]]))
    assert probs.shape == (1, 3, 4)


def test_sequence_step_loop_general():
    rng = torch.Generator()
    rng.manual_seed(0)

    logits = torch.tensor([-math.inf, 0.5, 0.5, -math.inf], dtype=torch.double)
    model = ModelStep(MockTokenizer(None), logits)

    sequence = Sequence(model)
    input_ids = torch.tensor([[1, 2, 1], [3, 4, 1]])
    token_ids, *_ = sequence.step(rng, 3, input_ids, torch.ones((1, 3)), samples=3)
    result, *_ = sequence.step(rng, 3, token_ids, torch.ones((3, 4)))
    assert result.shape == (1, 3, 2, 1)
    assert torch.equal(
        result.squeeze(0),
        torch.tensor(
            [
                [[1], [2]],
                [[1], [2]],
                [[1], [1]],
            ]
        ),
    )


def test_call_single_prompt():
    class FinishAfterTwo(Sequence):
        def __init__(self, model):
            super().__init__(model)
            self.iteration_idx = 0

        def is_finished(self, token_ids):
            """Finish generating the sequence after two iterations"""
            if self.iteration_idx == 0:
                self.iteration_idx += 1
                return torch.tensor([False])
            else:
                return torch.tensor([True])

    tokenizer = MockTokenizer({"Test": 0, "a": 1, "b": 2})
    model = MockModel(
        tokenizer,
        torch.tensor([[1.0, -math.inf, -math.inf], [-math.inf, 1.0, -math.inf]]),
    )
    sequence = FinishAfterTwo(model)

    result = sequence("Test")
    assert result == "Testa"


def test_call_prompt_list():
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
                return torch.tensor([False, False, False])
            elif self.iteration_idx == 1:
                self.iteration_idx += 1
                return torch.tensor([True, False, True])
            else:
                return torch.tensor([True])  # We only consider the unfinished sequences

    tokenizer = MockTokenizer(
        {"Test1": 0, "Test2": 1, "a": 2, "b": 3, "c": 4, "Test3": 5}
    )
    model = MockModel(
        tokenizer,
        torch.tensor(
            [
                [-math.inf, -math.inf, 1.0, -math.inf, -math.inf, -math.inf],
                [-math.inf, -math.inf, -math.inf, 1.0, -math.inf, -math.inf],
                [-math.inf, -math.inf, -math.inf, -math.inf, 1.0, -math.inf],
            ]
        ),
    )
    sequence = FinishAfterThree(model)

    result = sequence(["Test1", "Test2", "Test3"])
    assert result == ["ab<pad>", "abc", "ab<pad>"]


def test_call_single_prompt_samples():
    class FinishAfterTwo(Sequence):
        def __init__(self, model):
            super().__init__(model)
            self.iteration_idx = 0

        def is_finished(self, token_ids):
            if self.iteration_idx == 0:
                self.iteration_idx += 1
                return torch.tensor([False, False, False])
            else:
                return torch.tensor([True, True, True])

    tokenizer = MockTokenizer({"a": 0, "b": 1, "c": 2, "Test": 4})
    model = MockModel(
        tokenizer,
        torch.tensor(
            [
                [1, -math.inf, -math.inf, -math.inf],
                [-math.inf, 1, -math.inf, -math.inf],
            ],
            dtype=torch.double,
        ),
    )
    sequence = FinishAfterTwo(model)
    result = sequence("Test", samples=3)
    assert result == ["ab", "ab", "ab"]

    class FinishAfterOne(Sequence):
        def __init__(self, model):
            super().__init__(model)

        def is_finished(self, token_ids):
            return torch.tensor([True, True, True])

    tokenizer = MockTokenizer({"a": 0, "b": 1, "c": 3, "Test": 4})
    model = MockModel(
        tokenizer,
        torch.tensor(
            [
                [1, -math.inf, -math.inf, -math.inf],
                [-math.inf, 1, -math.inf, -math.inf],
            ],
            dtype=torch.double,
        ),
    )
    sequence = FinishAfterOne(model)
    result = sequence("Test", samples=3)
    assert result == ["a", "a", "a"]


def test_call_prompt_list_samples():
    class FinishAfterThree(Sequence):
        def __init__(self, model):
            super().__init__(model)
            self.iteration_idx = 0

        def is_finished(self, token_ids):
            if self.iteration_idx == 0:
                self.iteration_idx += 1
                batch_shape = token_ids.shape[:-1]
                return torch.zeros(batch_shape, dtype=torch.bool)
            elif self.iteration_idx == 1:
                self.iteration_idx += 1
                return torch.tensor(
                    [[True, False, True], [True, False, True], [True, False, True]]
                )
            else:
                return torch.tensor([True, True, True])

    tokenizer = MockTokenizer(
        {"a": 0, "b": 1, "c": 2, "Test1": 3, "Test2": 4, "Test3": 5}
    )
    model = MockModel(
        tokenizer,
        torch.tensor(
            [
                [1, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                [-math.inf, 1, -math.inf, -math.inf, -math.inf, -math.inf],
                [-math.inf, -math.inf, 1, -math.inf, -math.inf, -math.inf],
            ],
            dtype=torch.double,
        ),
    )
    sequence = FinishAfterThree(model)

    result = sequence(["Test1", "Test2", "Test3"], samples=3)
    assert result == [
        "ab<pad>",
        "abc",
        "ab<pad>",
        "ab<pad>",
        "abc",
        "ab<pad>",
        "ab<pad>",
        "abc",
        "ab<pad>",
    ]
