"""Base class for logits processors."""

from abc import abstractmethod
from typing import TypeVar

from outlines.processors.tensor_adapters import (
    TensorAdapterImplementation,
    tensor_adapters,
)

TensorType = TypeVar('TensorType')


class OutlinesLogitsProcessor:
    """Base class for logits processors.
    This class implements a shared `__call__` method is called by the models
    and returns the processed logits. It relies on the `process_logits` method
    that must be implemented by the subclasses to do the actual processing. The
    `tensor_adapter` attribute, created at initialization based on the
    tensor library name specified in the constructor, is used to manipulate the
    tensors using the appropriate library for the model (numpy, torch...).
    """
    tensor_adapter: TensorAdapterImplementation

    def __init__(self, tensor_library_name: str):
        """
        Parameters
        ----------
        tensor_library_name
            The name of the library to use to manipulate tensors. Possible
            values are "mlx", "numpy" and "torch". You must choose the library
            that your model is using.
        """
        # Temporary fix as torch raises a warning that can cause can an error
        # with python 3.12.
        if tensor_library_name == "torch":
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True

        tensor_adapter_class = tensor_adapters.get(tensor_library_name)
        if tensor_adapter_class is None:
            raise NotImplementedError(
                f"Library {tensor_library_name} is not available"
            )
        self.tensor_adapter = tensor_adapter_class()  # type: ignore

    def reset(self):
        """Reset the logits processor for a new generation

        Only implement this method in subclasses if the logits processor
        needs to be reset for a new generation.

        """
        pass # pragma: no cover

    @abstractmethod
    def process_logits(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Main method to implement for logits processors subclasses.
        This method applies a mask on the logits to bias the generation.
        It is called by the `__call__` method that standardizes the shape of
        `input_ids` and `logits` to ensure they are 2D tensors.
        Elements to keep in mind when designing universal logits processors:
        - logits processors are only used once and never re-applied for a new
        sequence generator
        - Some models only pass output_ids, some models such as llamacpp and
        transformers prefix with input_ids
        - Some sampling methods, such as beam search, result in unstable
        sequence ordering in models like vLLM
        Parameters
        ----------
        input_ids
            The ids of the tokens of the existing sequences in a 2D tensor.
        logits
            The logits for the current generation step in a 2D tensor.
        Returns
        -------
        TensorType
            The processed logits as a 2D tensor.
        """
        ...

    def __call__(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Entrypoint for logits processors, this is the method that is
        called by the model.
        Because different models use different structures to store the
        input_ids and logits, we standardize their format to 2D tensors
        before calling the `process_logits` method. After processing, the
        logits are cast back to the original array library type before being
        returned.
        Parameters
        ----------
        input_ids
            The ids of the tokens of the existing sequences in a tensor.
        logits
            The logits for the current generation step in a tensor.
        Returns
        -------
        TensorType
            The processed logits as a tensor.
        """
        # if input_ids is 1D and logits is 2D with a single sequence,
        # reshape input_ids to 2D (needed for mlx-lm)
        if (
            len(self.tensor_adapter.shape(input_ids)) == 1
            and len(self.tensor_adapter.shape(logits)) == 2
            and self.tensor_adapter.shape(logits)[0] == 1
        ):
            input_ids = self.tensor_adapter.unsqueeze(input_ids)

        assert (
            self.tensor_adapter.shape(logits)[:-1]
            == self.tensor_adapter.shape(input_ids)[:-1]
        )

        # Guarantee passed as 2D Tensors, then covert back to original
        # (1D or 2D) shape
        if len(self.tensor_adapter.shape(logits)) == 2:
            processed_logits = self.process_logits(input_ids, logits)
        elif len(self.tensor_adapter.shape(logits)) == 1:
            processed_logits = self.tensor_adapter.squeeze(
                self.process_logits(
                    self.tensor_adapter.unsqueeze(input_ids),
                    self.tensor_adapter.unsqueeze(logits),
                ),
            )
        else:
            raise ValueError(
                f"Logits shape {self.tensor_adapter.shape(logits)} is not "
                + "supported"
            )

        return processed_logits
