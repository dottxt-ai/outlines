"""Tensor adapter for the `mlx` library."""

from outlines.processors.tensor_adapters.base import TensorAdapter


class MLXTensorAdapter(TensorAdapter):
    library_name = "mlx"

    def __init__(self):
        import mlx.core

        self.mlx = mlx.core

    def shape(self, tensor):
        return tensor.shape

    def unsqueeze(self, tensor):
        return self.mlx.expand_dims(tensor, 0)

    def squeeze(self, tensor):
        if tensor.shape[0] == 1:
            return tensor[0]
        return tensor

    def to_list(self, tensor):
        return tensor.tolist()

    def to_scalar(self, tensor):
        return tensor.item()

    def full_like(self, tensor, fill_value):
        # Compatible with receiving a torch tensor
        return self.mlx.full(tensor.shape, fill_value)

    def concatenate(self, tensors):
        # Can handle both torch and mlx tensors
        return self.mlx.concatenate(
            [
                self.mlx.array(t) if not isinstance(t, self.mlx.array) else t
                for t in tensors
            ],
            axis=0
        )

    def get_device(self, tensor):
        return None

    def to_device(self, tensor, device):
        return tensor

    def boolean_ones_like(self, tensor):
        return self.mlx.ones(tensor.shape, dtype=self.mlx.bool_)

    def apply_mask(self, tensor, mask, value):
        result = tensor.astype(tensor.dtype)
        result = self.mlx.where(mask, self.mlx.array(value), result)
        return result

    def argsort_descending(self, tensor):
        return self.mlx.argsort(-tensor)
