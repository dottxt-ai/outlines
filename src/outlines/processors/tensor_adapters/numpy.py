"""Tensor adapter for the `numpy` library."""

from outlines.processors.tensor_adapters.base import TensorAdapter


class NumpyTensorAdapter(TensorAdapter):
    library_name = "numpy"

    def __init__(self):
        import numpy

        self.numpy = numpy

    def shape(self, tensor):
        return tensor.shape

    def unsqueeze(self, tensor):
        return self.numpy.expand_dims(tensor, axis=0)

    def squeeze(self, tensor):
        return self.numpy.squeeze(tensor, axis=0)

    def to_list(self, tensor):
        return tensor.tolist()

    def to_scalar(self, tensor):
        return tensor.item()

    def full_like(self, tensor, fill_value):
        return self.numpy.full_like(tensor, fill_value)

    def concatenate(self, tensors):
        return self.numpy.concatenate(tensors, axis=0)

    def get_device(self, tensor):
        return None

    def to_device(self, tensor, device):
        return tensor

    def boolean_ones_like(self, tensor):
        return self.numpy.ones_like(tensor, dtype=bool)

    def apply_mask(self, tensor, mask, value):
        result = tensor.copy()
        result[mask] = value
        return result

    def argsort_descending(self, tensor):
        return self.numpy.argsort(-tensor)
