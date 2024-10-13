from typing import List

import jax.numpy as jnp
import jaxlib
import numpy as np
import pytest
import torch

# Import mlx and the processor class
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from outlines.processors.base_logits_processor import (
    OutlinesLogitsProcessor,
    is_jax_array_type,
    is_mlx_array_type,
)


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


# ---- Test _to_torch() ---- #
def test_to_torch_with_numpy(processor):
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    torch_tensor = processor._to_torch(np_array)
    assert isinstance(torch_tensor, torch.Tensor)
    assert torch.allclose(
        torch_tensor, torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    )


def test_to_torch_with_list(processor):
    data_list = [[1, 2], [3, 4]]
    torch_tensor = processor._to_torch(data_list)
    assert isinstance(torch_tensor, torch.Tensor)
    assert torch.allclose(
        torch_tensor, torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    )


def test_to_torch_with_jax(processor):
    jax_array = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    torch_tensor = processor._to_torch(jax_array)
    assert isinstance(torch_tensor, torch.Tensor)
    assert torch.allclose(
        torch_tensor.cpu(), torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    )


@pytest.mark.skipif(not MLX_AVAILABLE, reason="mlx is not installed")
def test_to_torch_with_mlx(processor):
    mlx_array = mx.array([[1, 2], [3, 4]], dtype=np.float32)
    torch_tensor = processor._to_torch(mlx_array)
    assert isinstance(torch_tensor, torch.Tensor)
    assert torch.allclose(
        torch_tensor, torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    )


# ---- Test _from_torch() ---- #
def test_from_torch_to_numpy(processor):
    torch_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    np_array = processor._from_torch(torch_tensor, np.ndarray)
    assert isinstance(np_array, np.ndarray)
    assert np.allclose(np_array, np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_from_torch_to_list(processor):
    torch_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    data_list = processor._from_torch(torch_tensor, list)
    assert isinstance(data_list, list)
    assert data_list == [[1, 2], [3, 4]]


def test_from_torch_to_jax(processor):
    torch_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    jax_array = processor._from_torch(torch_tensor, jaxlib.xla_extension.ArrayImpl)
    print(f"JAX TYPE: {is_jax_array_type(type(jax_array))}")
    assert is_jax_array_type(type(jax_array))
    assert np.allclose(
        np.array(jax_array), np.array([[1, 2], [3, 4]], dtype=np.float32)
    )


@pytest.mark.skipif(not MLX_AVAILABLE, reason="mlx is not installed")
def test_from_torch_to_mlx(processor):
    torch_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    mlx_array = processor._from_torch(torch_tensor, mx.array)
    assert is_mlx_array_type(type(mlx_array))
    assert np.allclose(
        np.array(mlx_array), np.array([[1, 2], [3, 4]], dtype=np.float32)
    )


# ---- Test process_logits ---- #
def test_process_logits(processor):
    input_ids = [[1, 2], [3, 4]]
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    processed_logits = processor.process_logits(input_ids, logits)

    # multiplied by 2 as per mock implementation
    assert torch.allclose(
        processed_logits, torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
    )


# ---- Test __call__() ---- #
def test_call_with_numpy(processor):
    input_ids = np.array([[1, 2], [3, 4]], dtype=np.int32)
    logits = np.array([[0.5, 0.2], [0.1, 0.9]], dtype=np.float32)
    processed_logits = processor(input_ids, logits)

    # utput should match the mock behavior (logits * 2) and be in the original format (numpy)
    assert isinstance(processed_logits, np.ndarray)
    assert np.allclose(
        processed_logits, np.array([[1.0, 0.4], [0.2, 1.8]], dtype=np.float32)
    )


def test_call_with_jax(processor):
    input_ids = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    logits = jnp.array([[0.5, 0.2], [0.1, 0.9]], dtype=jnp.float32)
    processed_logits = processor(input_ids, logits)

    # output should match the mock behavior (logits * 2) and be in the original format (JAX array)
    assert is_jax_array_type(type(processed_logits))
    assert np.allclose(
        np.array(processed_logits), np.array([[1.0, 0.4], [0.2, 1.8]], dtype=np.float32)
    )
