from typing import Union

from .jax import JAXTensorAdapter
from .mlx import MLXTensorAdapter
from .numpy import NumpyTensorAdapter
from .torch import TorchTensorAdapter


tensor_adapters = {
    "jax": JAXTensorAdapter,
    "mlx": MLXTensorAdapter,
    "numpy": NumpyTensorAdapter,
    "torch": TorchTensorAdapter,
}

TensorAdapterImplementation = Union[
    JAXTensorAdapter,
    MLXTensorAdapter,
    NumpyTensorAdapter,
    TorchTensorAdapter,
]
