from abc import ABC, abstractmethod
from typing import Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class TensorAdapter(ABC):
    """Abstract base class for tensor adapters.

    This class defines the interface for tensor adapters that are used to
    manipulate tensors in different libraries. Concrete implementations of
    this class should provide specific implementations for each method as
    well as providing a library_name attribute.

    TODO: Update the version of outlines-core used to receive plain arrays
    instead of torch tensors. In the meantime, implementations of this class
    must make sure that their `full_like` and `concatenate` methods can
    handle torch tensors.
    """
    library_name: str

    @abstractmethod
    def shape(self, tensor):
        """Get the shape of the tensor"""
        ...

    @abstractmethod
    def unsqueeze(self, tensor):
        """Add a dimension to the tensor at the specified position"""
        ...

    @abstractmethod
    def squeeze(self, tensor):
        """Remove a dimension from the tensor at the specified position"""
        ...

    @abstractmethod
    def to_list(self, tensor):
        """Convert the tensor to a list"""
        ...

    @abstractmethod
    def to_scalar(self, tensor):
        """Convert the tensor to a scalar"""
        ...

    @abstractmethod
    def full_like(self, tensor: "torch.Tensor", fill_value):
        """Create a tensor with the same shape as the input tensor filled
        with a scalar value.
        ATTENTION: This method receives a torch tensor regardless of the
        library used.
        """
        ...

    @abstractmethod
    def concatenate(self, tensors: list[Union["torch.Tensor", Any]]):
        """Concatenate a list of tensors along a specified dimension.
        ATTENTION: This method can either receive a list of torch tensors or
        a list of tensors from the library used.
        """
        ...

    @abstractmethod
    def get_device(self, tensor):
        """Get the device of the tensor"""
        ...

    @abstractmethod
    def to_device(self, tensor, device):
        """Move the tensor to a specified device"""
        ...

    @abstractmethod
    def boolean_ones_like(self, tensor):
        """Create a boolean ones tensor with the same shape as the input tensor"""
        ...

    @abstractmethod
    def apply_mask(self, tensor, mask, value):
        """Fill the elements of the tensor where the mask is True with the specified value"""
        ...

    @abstractmethod
    def argsort_descending(self, tensor):
        """Return the indices that would sort the tensor in descending order"""
        ...
