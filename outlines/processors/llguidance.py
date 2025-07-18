"""Logits processor using LLGuidance to bias the logits."""

import warnings

from llguidance import LLMatcher

from outlines.processors.base_logits_processor import OutlinesLogitsProcessor, TensorType


# TODO: add tensorflow and jax
SUPPORTED_TENSOR_LIBRARIES = ["numpy", "mlx", "torch"]


class LLGuidanceLogitsProcessor(OutlinesLogitsProcessor):
    """Logits Processor for the LLGuidance backend."""

    def __init__(
        self,
        grammar: str,
        llg_tokenizer,
        tensor_library_name: str,
    ) -> None:
        """
        Parameters
        ----------
        grammar: str
            The grammar spec to use to create the LLMatcher
        llg_tokenizer: LLTokenizer
            The LLGuidance tokenizer
        tensor_library_name: str
            The name of the tensor library used by the model

        """
        if tensor_library_name not in SUPPORTED_TENSOR_LIBRARIES:
            raise TypeError(f"Unsupported tensor library: {tensor_library_name}")

        self.is_first_token = True
        self.grammar = grammar
        self.llg_tokenizer = llg_tokenizer
        self.tensor_library_name = tensor_library_name
        super().__init__(tensor_library_name)

    def _setup(self, batch_size: int) -> None:
        """Setup the LLMatchers, the bitmask and some functions used in the
        `process_logits` method.

        This method is called when the first token is generated instead of
        at initialization because we need to know the batch size.

        Parameters
        ----------
        batch_size: int
            The batch size of the input

        """
        self.ll_matchers = [
            LLMatcher(self.llg_tokenizer, self.grammar)
            for _ in range(batch_size)
        ]

        # we must adapt the bitmask creation and the bias function to the
        # tensor library used by the model
        if self.tensor_library_name == "torch":
            import llguidance.torch

            self.bitmask = llguidance.torch.allocate_token_bitmask(batch_size, self.llg_tokenizer.vocab_size)
            self._bias_logits = self._bias_logits_torch
        elif self.tensor_library_name == "numpy":
            import llguidance.numpy

            self.bitmask = llguidance.numpy.allocate_token_bitmask(batch_size, self.llg_tokenizer.vocab_size)
            self._bias_logits = self._bias_logits_numpy
        elif self.tensor_library_name == "mlx":
            import llguidance.numpy

            self.bitmask = llguidance.numpy.allocate_token_bitmask(batch_size, self.llg_tokenizer.vocab_size)
            self._bias_logits = self._bias_logits_mlx
        else:
            raise ValueError(f"Unsupported tensor library: {self.tensor_library_name}")

    def _bias_logits_mlx(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for the MLX backend."""
        import llguidance.mlx
        import llguidance.numpy

        biased_logits_array = []
        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            llguidance.numpy.fill_next_token_bitmask(self.ll_matchers[i], self.bitmask, i)
            biased_logits = llguidance.mlx.apply_token_bitmask(
                logits[i], self.bitmask[i] # type: ignore
            )
            biased_logits_array.append(biased_logits)

        return self.tensor_adapter.concatenate(biased_logits_array)

    def _bias_logits_torch(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for the Torch backend."""
        import llguidance.torch

        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            llguidance.torch.fill_next_token_bitmask(self.ll_matchers[i], self.bitmask, i)
            llguidance.torch.apply_token_bitmask_inplace(
                logits[i], self.bitmask[i] # type: ignore
            )

        return logits

    def _bias_logits_numpy(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for the Numpy backend."""
        import llguidance.numpy

        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            llguidance.numpy.fill_next_token_bitmask(self.ll_matchers[i], self.bitmask, i)
            llguidance.numpy.apply_token_bitmask_inplace(
                logits[i], self.bitmask[i] # type: ignore
            )

        return logits

    def process_logits(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Use the instances of LLMatcher to bias the logits.

        Parameters
        ----------
        input_ids
            The ids of the tokens of the existing sequences.
        logits
            The logits for the current generation step.

        Returns
        -------
        TensorType
            The biased logits.

        """
        if self.is_first_token:
            self._setup(self.tensor_adapter.shape(input_ids)[0])
            self.is_first_token = False

        # we do not make the matchers consume the last token during the first
        # generation step because no tokens have been generated yet
        else:
            for i in range(self.tensor_adapter.shape(input_ids)[0]):
                sequence = input_ids[i] # type: ignore
                last_token = sequence[-1].item()
                self.ll_matchers[i].consume_token(last_token)
                error = self.ll_matchers[i].get_error()
                if error:
                    warnings.warn(f"Error in LLMatcher: {error}")

        return self._bias_logits(input_ids, logits)
