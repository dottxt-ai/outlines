"""Library specific objects to manipulate tensors."""

from typing import Union

from .jax import JAXTensorAdapter
from .mlx import MLXTensorAdapter
from .numpy import NumpyTensorAdapter
from .tensorflow import TensorFlowTensorAdapter
from .torch import TorchTensorAdapter


tensor_adapters = {
    "jax": JAXTensorAdapter,
    "mlx": MLXTensorAdapter,
    "numpy": NumpyTensorAdapter,
    "tensorflow": TensorFlowTensorAdapter,
    "torch": TorchTensorAdapter,
}

TensorAdapterImplementation = Union[
    JAXTensorAdapter,
    MLXTensorAdapter,
    NumpyTensorAdapter,
    TensorFlowTensorAdapter,
    TorchTensorAdapter,
]
