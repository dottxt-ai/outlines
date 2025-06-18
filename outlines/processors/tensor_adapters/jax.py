"""Tensor adapter for the `jax` library."""

from outlines.processors.tensor_adapters.base import TensorAdapter


class JAXTensorAdapter(TensorAdapter):
    library_name = "jax"

    def __init__(self):
        import jax

        self.jax = jax

    def shape(self, tensor):
        return tensor.shape

    def unsqueeze(self, tensor):
        return self.jax.numpy.expand_dims(tensor, axis=0)

    def squeeze(self, tensor):
        return self.jax.numpy.squeeze(tensor, axis=0)

    def to_list(self, tensor):
        return tensor.tolist()

    def to_scalar(self, tensor):
        return tensor.item()

    def full_like(self, tensor, fill_value):
        return self.jax.numpy.full_like(tensor, fill_value)

    def concatenate(self, tensors):
        return self.jax.numpy.concatenate(tensors, axis=0)

    def to_device(self, tensor, device):
        return tensor

    def get_device(self, tensor):
        return None

    def boolean_ones_like(self, tensor):
        return self.jax.numpy.ones_like(tensor, dtype=bool)

    def apply_mask(self, tensor, mask, value):
        result = tensor.copy()
        result = result.at[mask].set(value)
        return result

    def argsort_descending(self, tensor):
        return self.jax.numpy.argsort(-tensor)
