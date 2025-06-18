"""Base class for tensor adapters."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Any, Union

if TYPE_CHECKING:
    import torch

TensorType = TypeVar('TensorType')


class TensorAdapter(ABC):
    """Abstract base class for tensor adapters.

    This class defines the interface for tensor adapters that are used to
    manipulate tensors in different libraries. Concrete implementations of
    this class should provide specific implementations for each method as
    well as providing a `library_name` attribute.

    TODO: Update the version of outlines-core used to receive plain arrays
    instead of torch tensors. In the meantime, implementations of this class
    must make sure that their `full_like` and `concatenate` methods can
    handle torch tensors.

    """
    library_name: str

    @abstractmethod
    def shape(self, tensor: TensorType) -> list[int]:
        """Get the shape of the tensor.

        Parameters
        ----------
        tensor
            The tensor to get the shape of.

        Returns
        -------
        list[int]
            The shape of the tensor. The list contains as many elements as
            there are dimensions in the tensor.

        """
        ...

    @abstractmethod
    def unsqueeze(self, tensor: TensorType) -> TensorType:
        """Add a dimension to the tensor at axis 0.

        Parameters
        ----------
        tensor
            The tensor to add a dimension to.

        Returns
        -------
        TensorType
            The tensor with an additional dimension.

        """
        ...

    @abstractmethod
    def squeeze(self, tensor: TensorType) -> TensorType:
        """Remove a dimension from the tensor at axis 0.

        Parameters
        ----------
        tensor
            The tensor to remove a dimension from.

        Returns
        -------
        TensorType
            The tensor with one less dimension.

        """
        ...

    @abstractmethod
    def to_list(self, tensor: TensorType) -> list:
        """Convert the tensor to a list.

        Parameters
        ----------
        tensor
            The tensor to convert to a list.

        Returns
        -------
        list
            The tensor as a list.

        """
        ...

    @abstractmethod
    def to_scalar(self, tensor: TensorType) -> Any:
        """Return the only element of the tensor.

        Parameters
        ----------
        tensor
            The tensor to return the only element of.

        Returns
        -------
        Any
            The only element of the tensor.

        """
        ...

    @abstractmethod
    def full_like(self, tensor: "torch.Tensor", fill_value: Any) -> TensorType: # type: ignore
        """Create a tensor with the same shape as the input tensor filled
        with a scalar value.

        ATTENTION: This method receives a torch tensor regardless of the
        library used.

        Parameters
        ----------
        tensor
            The tensor to create a new tensor with the same shape.
        fill_value
            The value to fill the new tensor with.

        Returns
        -------
        TensorType
            A tensor with the same shape as the input tensor filled with the
            specified value.

        """
        ...

    @abstractmethod
    def concatenate(
        self, tensors: list[Union["torch.Tensor", TensorType]]
    ) -> TensorType:
        """Concatenate a list of tensors along axis 0.

        ATTENTION: This method can either receive a list of torch tensors or
        a list of tensors from the library used.

        Parameters
        ----------
        tensors
            The list of tensors to concatenate.

        Returns
        -------
        TensorType
            The concatenated tensor.

        """
        ...

    @abstractmethod
    def get_device(self, tensor: TensorType) -> str:
        """Get the name of the tensor's device.

        Parameters
        ----------
        tensor
            The tensor to get the device of.

        Returns
        -------
        str
            The name of the tensor's device.

        """
        ...

    @abstractmethod
    def to_device(self, tensor: TensorType, device: str) -> TensorType:
        """Move the tensor to a specified device.

        Parameters
        ----------
        tensor
            The tensor to move to a specified device.
        device
            The name of the device to move the tensor to.

        Returns
        -------
        TensorType
            The tensor moved to the specified device.

        """
        ...

    @abstractmethod
    def boolean_ones_like(self, tensor: TensorType) -> TensorType:
        """Create a boolean ones tensor with the same shape as the input
        tensor.

        Parameters
        ----------
        tensor
            The tensor to create a boolean ones tensor with the same shape.

        Returns
        -------
        TensorType
            A boolean ones tensor with the same shape as the input tensor.

        """
        ...

    @abstractmethod
    def apply_mask(
        self, tensor: TensorType, mask: TensorType, value: Any
    ) -> TensorType:
        """Fill the elements of the tensor where the mask is True with the
        specified value.

        Parameters
        ----------
        tensor
            The tensor to fill.
        mask
            The mask to apply to the tensor.
        value
            The value to fill the tensor with.

        Returns
        -------
        TensorType
            The tensor with the mask applied.

        """
        ...

    @abstractmethod
    def argsort_descending(
        self, tensor: TensorType
    ) -> TensorType:
        """Return the indices that would sort the tensor in descending order
        along axis -1.

        Parameters
        ----------
        tensor
            The tensor to sort.

        Returns
        -------
        TensorType
            The indices that would sort the tensor in descending order along
            axis -1.

        """
        ...
