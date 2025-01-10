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
    assert np.allclose(data, arrays[array_type])


@pytest.mark.parametrize("array_type", arrays.keys())
def test_call(array_type, processor):
    input_ids = arrays[array_type]
    logits = arrays[array_type]
    processed_logits = processor(input_ids, logits)

    assert isinstance(processed_logits, type(arrays[array_type]))
    assert np.allclose(
        np.array(processed_logits), np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    )
