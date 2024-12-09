from abc import abstractmethod
from typing import TYPE_CHECKING, List, Protocol, Type, Union

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    import mlx.core as mx


Array = Union[NDArray, torch.Tensor, List, "mx.array"]


def is_mlx_array_type(array_type):
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return issubclass(array_type, mx.array)


def is_jax_array_type(array_type):
    try:
        import jaxlib
    except ImportError:
        return False
    return issubclass(array_type, jaxlib.xla_extension.ArrayImpl) or isinstance(
        array_type, jaxlib.xla_extension.ArrayImpl
    )


class OutlinesLogitsProcessor(Protocol):
    """
    Base class for logits processors which normalizes types of logits:
    - ndarray (used by llama-cpp-python), converted to torch.Tensor
    - mlx.core.array (used by mlx-lm), converted to torch.Tensor
    - torch.Tensor (used by everything else)

    Normalization of types and conversion to torch.Tensor
    doesn't move memory, it just casts the type.

    Normalizing the types allows all logits processors inheriting from this class
    to implement a single method for all the business logit: `process_logits()`
    """

    @abstractmethod
    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids and logits are always 2D tensors for handling a batch of sequences.

        - input_ids -> List[List[tokens]]
        - logits -> 2D_Tensor[logit floats]

        Important to keep in mind when designing universal logits processors
        - logits processors are only used once and never re-applied for a new sequence generator
        - Some models only pass output_ids, some models such as llamacpp and transformers prefix with input_ids
        - Some sampling methods, such as beam search, result in unstable sequence ordering in models like vLLM
        """
        pass

    @torch.no_grad()
    def __call__(
        self,
        input_ids: Array,
        logits: Array,
    ) -> Array:
        """
        Apply logits processor

        1) Unify type
        - convert input_ids: either ndarray, mlx array, List[int], or Tensor -> List[List[int]]
        - convert logits: either ndarray, mlx array, or Tensor -> 2D float Tensor
        2) Unify shape, ensure logits and input_ids are 2D
        3) Call self.process_logits() to perform business logic
        4) Cast logits back to original array library type
        """
        # ensure logits are torch Tensors
        torch_logits = self._to_torch(logits)
        input_ids = self._to_torch(input_ids)

        assert torch_logits.shape[:-1] == input_ids.shape[:-1]

        # Guarantee passed as 2D Tensors, then covert back to original (1D or 2D) shape
        if len(torch_logits.shape) == 2:
            processed_logits = self.process_logits(input_ids, torch_logits)
        elif len(torch_logits.shape) == 1:
            processed_logits = self.process_logits(
                input_ids.unsqueeze(0), torch_logits.unsqueeze(0)
            ).squeeze(0)

        # return logits as passed array type
        return self._from_torch(processed_logits, type(logits))

    @staticmethod
    def _to_torch(tensor_like: Array) -> torch.Tensor:
        """Convert various types to torch.Tensor."""
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like

        elif isinstance(tensor_like, np.ndarray):
            return torch.from_numpy(tensor_like)

        elif isinstance(tensor_like, (list, tuple)):
            return torch.tensor(tensor_like)

        elif is_mlx_array_type(type(tensor_like)):
            import mlx.core as mx

            # https://ml-explore.github.io/mlx/build/html/usage/numpy.html#pytorch
            return torch.from_dlpack(
                np.array(tensor_like.astype(mx.float32), copy=False)
            )

        elif is_jax_array_type(type(tensor_like)):
            import jax

            torch_tensor = torch.from_dlpack(jax.dlpack.to_dlpack(tensor_like))
            return torch_tensor

        else:
            raise TypeError(
                "LogitsProcessor must be called with either np.NDArray, "
                "torch.Tensor, list, or mlx.core.array typed logits. "
                f"Logits type: `{type(tensor_like)}`"
            )

    @staticmethod
    def _from_torch(tensor: torch.Tensor, target_type: Type) -> Array:
        """Convert torch.Tensor to the specified target type."""
        if target_type == torch.Tensor:
            return tensor

        elif target_type == np.ndarray:
            return tensor.detach().numpy()

        elif target_type == list:
            return tensor.detach().tolist()

        elif target_type == tuple:
            return tuple(tensor.detach().tolist())

        elif is_mlx_array_type(target_type):
            import mlx.core as mx

            # numpy doesn't support bfloat16, mlx doesn't support direct conversion from torch
            return mx.array(tensor.float().numpy())

        elif is_jax_array_type(target_type):
            import jax

            return jax.dlpack.from_dlpack(tensor)

        else:
            raise TypeError(
                f"Failed to convert torch tensors to target_type `{target_type}`"
            )
