from unittest.mock import patch

import numpy as np
import torch
from outlines_core import Guide, Index, Vocabulary

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor


def make_index():
    vocabulary = Vocabulary(4, {"a": [0], "b": [1], "ab": [2], "x": [3]})
    return Index("(a|b)*", vocabulary)


def test_torch_processor_applies_batch_mask_once():
    from outlines_core.kernels.torch import (
        allocate_token_bitmask,
        apply_token_bitmask_inplace,
        fill_next_token_bitmask,
    )

    index = make_index()
    processor = OutlinesCoreLogitsProcessor(index, "torch")
    guides = [Guide(index) for _ in range(3)]
    masks = [allocate_token_bitmask(5) for _ in guides]
    input_ids = torch.tensor([[0], [1], [2]])

    for step in range(2):
        actual = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        expected = actual.clone()
        if step:
            for guide, token_id in zip(guides, input_ids[:, -1].tolist()):
                guide.advance(token_id, return_tokens=False)
        for i, (guide, mask) in enumerate(zip(guides, masks)):
            fill_next_token_bitmask(guide, mask)
            apply_token_bitmask_inplace(expected[i : i + 1], mask)

        with patch(
            "outlines_core.kernels.torch.apply_token_bitmask_inplace",
            wraps=apply_token_bitmask_inplace,
        ) as apply_mask:
            processor(input_ids, actual)

        assert apply_mask.call_count == 1
        assert apply_mask.call_args.args[1].shape[0] == 3
        assert torch.equal(actual, expected)


def test_numpy_processor_applies_batch_mask_once():
    from outlines_core.kernels.numpy import (
        allocate_token_bitmask,
        apply_token_bitmask_inplace,
        fill_next_token_bitmask,
    )

    index = make_index()
    processor = OutlinesCoreLogitsProcessor(index, "numpy")
    guides = [Guide(index) for _ in range(3)]
    masks = [allocate_token_bitmask(5) for _ in guides]
    input_ids = np.array([[0], [1], [2]])

    for step in range(2):
        actual = np.arange(15, dtype=np.float32).reshape(3, 5)
        expected = actual.copy()
        if step:
            for guide, token_id in zip(guides, input_ids[:, -1].tolist()):
                guide.advance(token_id, return_tokens=False)
        for i, (guide, mask) in enumerate(zip(guides, masks)):
            fill_next_token_bitmask(guide, mask)
            apply_token_bitmask_inplace(expected[i : i + 1], mask)

        with patch(
            "outlines_core.kernels.numpy.apply_token_bitmask_inplace",
            wraps=apply_token_bitmask_inplace,
        ) as apply_mask:
            processor(input_ids, actual)

        assert apply_mask.call_count == 1
        assert apply_mask.call_args.args[1].shape[0] == 3
        np.testing.assert_array_equal(actual, expected)
