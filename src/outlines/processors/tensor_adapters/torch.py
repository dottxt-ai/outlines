"""Tensor adapter for the `torch` library."""

from outlines.processors.tensor_adapters.base import TensorAdapter


class TorchTensorAdapter(TensorAdapter):
    library_name = "torch"

    def __init__(self):
        import torch

        self.torch = torch

    def shape(self, tensor):
        return tensor.shape

    def unsqueeze(self, tensor):
        return tensor.unsqueeze(0)

    def squeeze(self, tensor):
        return tensor.squeeze(0)

    def to_list(self, tensor):
        return tensor.tolist()

    def to_scalar(self, tensor):
        return tensor.item()

    def full_like(self, tensor, fill_value):
        return self.torch.full_like(tensor, fill_value)

    def concatenate(self, tensors):
        return self.torch.cat(tensors, dim=0)

    def get_device(self, tensor):
        return tensor.device

    def to_device(self, tensor, device):
        return tensor.to(device)

    def boolean_ones_like(self, tensor):
        return self.torch.ones_like(tensor, dtype=self.torch.bool)

    def apply_mask(self, tensor, mask, value):
        return self.torch.masked_fill(tensor, mask, value)

    def argsort_descending(self, tensor):
        return self.torch.argsort(tensor, descending=True)
