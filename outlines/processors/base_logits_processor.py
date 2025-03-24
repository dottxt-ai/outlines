from abc import abstractmethod
from typing import TypeVar

from outlines.processors.tensor_adapters import tensor_adapters, TensorAdapterImplementation

TensorType = TypeVar('TensorType')


class OutlinesLogitsProcessor:
    """Base class for logits processors.

    This class standardizes the shape of input_ids and logits received and
    returned by the logits processor. At initialization, it creates a
    tensor_adapter attribute which is used to manipulate the tensors
    using the appropriate library for the model (numpy, torch...).
    """
    tensor_adapter: TensorAdapterImplementation

    def __init__(self, tensor_library_name: str):
        tensor_adapter_class = tensor_adapters.get(tensor_library_name)
        if tensor_adapter_class is None:
            raise NotImplementedError(f"Library {tensor_library_name} is not available")
        self.tensor_adapter = tensor_adapter_class()  # type: ignore

    @abstractmethod
    def process_logits(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Main method to implement for logits processors.

        This method applies a mask on the logits to bias the generation.
        It is called by the __call__ method that standardizes the shape of
        input_ids and logits to ensure they are 2D tensors.

        Important to keep in mind when designing universal logits processors
        - logits processors are only used once and never re-applied for a new
        sequence generator
        - Some models only pass output_ids, some models such as llamacpp and
        transformers prefix with input_ids
        - Some sampling methods, such as beam search, result in unstable
        sequence ordering in models like vLLM
        """
        ...

    def __call__(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Entrypoint for logits processors, this is the method that is called
        by the model.

        Because different models use different libraries, we need to standardize
        the shape of input_ids and logits to ensure they are 2D tensors before
        calling the unified process_logits method. After processing, the logits
        are cast back to the original array library type before being returned.
        """
        # if input_ids is 1D and logits is 2D with a single sequence,
        # reshape input_ids to 2D (needed for mlx-lm)
        if (
            len(self.tensor_adapter.shape(input_ids)) == 1
            and len(self.tensor_adapter.shape(logits)) == 2
            and self.tensor_adapter.shape(logits)[0] == 1
        ):
            input_ids = self.tensor_adapter.unsqueeze(input_ids)

        assert self.tensor_adapter.shape(logits)[:-1] == self.tensor_adapter.shape(input_ids)[:-1]

        # Guarantee passed as 2D Tensors, then covert back to original (1D or 2D) shape
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
            raise ValueError(f"Logits shape {self.tensor_adapter.shape(logits)} is not supported")

        return processed_logits
