"""Library specific objects to manipulate tensors."""

from typing import Union

from .mlx import MLXTensorAdapter
from .numpy import NumpyTensorAdapter
from .torch import TorchTensorAdapter

tensor_adapters = {
    "mlx": MLXTensorAdapter,
    "numpy": NumpyTensorAdapter,
    "torch": TorchTensorAdapter,
}

TensorAdapterImplementation = Union[
    MLXTensorAdapter,
    NumpyTensorAdapter,
    TorchTensorAdapter,
]

__all__ = [
    "MLXTensorAdapter",
    "NumpyTensorAdapter",
    "TorchTensorAdapter",
    "tensor_adapters",
    "TensorAdapterImplementation",
]
