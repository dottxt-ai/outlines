from typing import List

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from outlines.processors.base_logits_processor import OutlinesLogitsProcessor

arrays = {
    "list": [[1.0, 2.0], [3.0, 4.0]],
    "np": np.array([[1, 2], [3, 4]], dtype=np.float32),
    "jax": jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
    "torch": torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
}

try:
    import mlx.core as mx

    arrays["mlx"] = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    arrays["mlx_bfloat16"] = mx.array([[1, 2], [3, 4]], dtype=mx.bfloat16)
except ImportError:
    pass

try:
    import jax.numpy as jnp

    arrays["jax"] = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
except ImportError:
    pass


# Mock implementation of the abstract class for testing
class MockLogitsProcessor(OutlinesLogitsProcessor):
    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        # For testing purposes, let's just return logits multiplied by 2
        return logits * 2


@pytest.fixture
def processor():
    """Fixture for creating an instance of the MockLogitsProcessor."""
    return MockLogitsProcessor()


@pytest.mark.parametrize("array_type", arrays.keys())
def test_to_torch(array_type, processor):
    data = arrays[array_type]
    torch_tensor = processor._to_torch(data)
    assert isinstance(torch_tensor, torch.Tensor)
    assert torch.allclose(
        torch_tensor.cpu(), torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    )


@pytest.mark.parametrize("array_type", arrays.keys())
def test_from_torch(array_type, processor):
    torch_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    data = processor._from_torch(torch_tensor, type(arrays[array_type]))
    assert isinstance(data, type(arrays[array_type]))
    if array_type == "mlx_bfloat16":
        # For bfloat16, we expect the output to be float32 due to the conversion
        assert data.dtype == mx.float32
        assert np.allclose(np.array(data), np.array([[1, 2], [3, 4]], dtype=np.float32))
    else:
        assert np.allclose(data, arrays[array_type])


@pytest.mark.parametrize("array_type", arrays.keys())
def test_call(array_type, processor):
    input_ids = arrays[array_type]
    logits = arrays[array_type]
    processed_logits = processor(input_ids, logits)

    assert isinstance(processed_logits, type(arrays[array_type]))

    # Convert to numpy array for comparison, handling PyTorch tensors properly
    if array_type == "torch":
        expected = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        actual = processed_logits.detach().cpu().numpy()
        assert np.allclose(actual, expected)
    else:
        assert np.allclose(
            np.array(processed_logits), np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        )
