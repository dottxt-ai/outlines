from typing import List

import numpy as np
import pytest
import torch

from outlines.processors.base_logits_processor import OutlinesLogitsProcessor

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


libraries = ["numpy", "torch"]
if HAS_MLX:
    libraries.append("mlx")

# we check the accepted shapes:
# - both 1D
# - both 2D
# - input_ids 1D and logits 2D with a single sequence
# we raise an error if the shapes are not accepted:
# - input_ids 2D and logits 1D
# - input_ids 1D and logits 2D, but with multiple sequences
# - both 3D
arrays = {
    "numpy": [
        (np.array([1, 2], dtype=np.float32), np.array([1, 2], dtype=np.int32), None),
        (np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[1, 2], [3, 4]], dtype=np.int32), None),
        (np.array([1, 2], dtype=np.float32), np.array([[1, 2]], dtype=np.int32), None),
        (np.array([[1, 2]], dtype=np.float32), np.array([1, 2], dtype=np.int32), AssertionError),
        (np.array([1, 2], dtype=np.float32), np.array([[1, 2], [3, 4]], dtype=np.int32), AssertionError),
        (np.array([[[1, 2]]], dtype=np.float32), np.array([[[1, 2]]], dtype=np.int32), ValueError),
    ],
    "torch": [
        (torch.tensor([1, 2], dtype=torch.float32), torch.tensor([1, 2], dtype=torch.int32), None),
        (torch.tensor([[1, 2], [3, 4]], dtype=torch.float32), torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), None),
        (torch.tensor([1, 2], dtype=torch.float32), torch.tensor([[1, 2]], dtype=torch.int32), None),
        (torch.tensor([[1, 2]], dtype=torch.float32), torch.tensor([1, 2], dtype=torch.int32), AssertionError),
        (torch.tensor([1, 2], dtype=torch.float32), torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), AssertionError),
        (torch.tensor([[[1, 2]]], dtype=torch.float32), torch.tensor([[[1, 2]]], dtype=torch.int32), ValueError),
    ],
}
if HAS_MLX:
    arrays["mlx"] = [
        (mx.array([1, 2], dtype=mx.float32), mx.array([1, 2], dtype=mx.int32), None),
        (mx.array([[1, 2], [3, 4]], dtype=mx.float32), mx.array([[1, 2], [3, 4]], dtype=mx.int32), None),
        (mx.array([1, 2], dtype=mx.float32), mx.array([[1, 2]], dtype=mx.int32), None),
        (mx.array([[1, 2]], dtype=mx.float32), mx.array([1, 2], dtype=mx.int32), AssertionError),
        (mx.array([1, 2], dtype=mx.float32), mx.array([[1, 2], [3, 4]], dtype=mx.int32), AssertionError),
        (mx.array([[[1, 2]]], dtype=mx.float32), mx.array([[[1, 2]]], dtype=mx.int32), ValueError),
    ]

class MockLogitsProcessor(OutlinesLogitsProcessor):
    def process_logits(self, input_ids, logits):
        # check that input_ids and logits received are 2D tensors
        assert len(self.tensor_adapter.shape(input_ids)) == 2
        assert len(self.tensor_adapter.shape(logits)) == 2
        return logits


@pytest.mark.parametrize("library", libraries)
def test_base_logits_processor_init(library):
    processor = MockLogitsProcessor(library)
    assert processor.tensor_adapter is not None
    with pytest.raises(NotImplementedError):
        processor = MockLogitsProcessor("foo")
        processor.reset()


@pytest.mark.parametrize("library", libraries)
def test_base_logits_processor_call(library):
    processor = MockLogitsProcessor(library)
    input_values = arrays[library]
    for input_value in input_values:
        input_ids, logits, expected_error = input_value
        if expected_error is not None:
            with pytest.raises(expected_error):
                processor(input_ids, logits)
        else:
            original_shape = processor.tensor_adapter.shape(logits)
            processed_logits = processor(input_ids, logits)
            # we check that the shape of logits is preserved
            assert processor.tensor_adapter.shape(processed_logits) == original_shape


@pytest.mark.parametrize("library", libraries)
def test_base_logits_processor_init_library_name(library):
    processor = MockLogitsProcessor(library)
    assert processor.tensor_adapter is not None
    with pytest.raises(NotImplementedError):
        processor = MockLogitsProcessor("foo")
