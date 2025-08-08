"""Backend class for LLGuidance."""

import warnings
from typing import TYPE_CHECKING

from outlines.backends.base import BaseBackend
from outlines.models import LlamaCpp, MLXLM, SteerableModel, Transformers
from outlines.processors.base_logits_processor import (
    OutlinesLogitsProcessor,
    TensorType
)

if TYPE_CHECKING:
    from llguidance import LLGTokenizer


SUPPORTED_TENSOR_LIBRARIES = ["numpy", "mlx", "torch"]


class LLGuidanceLogitsProcessor(OutlinesLogitsProcessor):
    """Logits Processor for the LLGuidance backend."""

    def __init__(
        self,
        grammar: str,
        llg_tokenizer,
        tensor_library_name: str,
        *,
        end_thinking_token_id: int | None,
        thinking_max_tokens: int | None,
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
        end_thinking_token_id: int | None
            The id of the end thinking token
        thinking_max_tokens: int | None
            The maximum number of tokens the model can think about

        """
        if tensor_library_name not in SUPPORTED_TENSOR_LIBRARIES:
            raise TypeError(f"Unsupported tensor library: {tensor_library_name}")

        self.is_first_token = True
        self.grammar = grammar
        self.llg_tokenizer = llg_tokenizer
        self.tensor_library_name = tensor_library_name
        self.end_thinking_token_id = end_thinking_token_id
        self.thinking_max_tokens = thinking_max_tokens or float("inf")
        super().__init__(tensor_library_name)

    def reset(self):
        """Ensure self._setup is called again for the next generation."""
        self.is_first_token = True

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
        from llguidance import LLMatcher

        self._ll_matchers = [
            LLMatcher(self.llg_tokenizer, self.grammar)
            for _ in range(batch_size)
        ]
        self._is_thinking = [self.end_thinking_token_id is not None] * batch_size
        self._generate_end_thinking_token = [False] * batch_size
        self._num_tokens_generated = 0

        # we must adapt the bitmask creation and the bias function to the
        # tensor library used by the model
        if self.tensor_library_name == "torch":
            import llguidance.torch

            self.allocate_token_bitmask = llguidance.torch.allocate_token_bitmask
            self._bias_logits = self._bias_logits_torch
        elif self.tensor_library_name == "numpy":
            import llguidance.numpy

            self.allocate_token_bitmask = llguidance.numpy.allocate_token_bitmask
            self._bias_logits = self._bias_logits_numpy
        elif self.tensor_library_name == "mlx": # pragma: no cover
            import llguidance.numpy

            self.allocate_token_bitmask = llguidance.numpy.allocate_token_bitmask
            self._bias_logits = self._bias_logits_mlx
        else: # pragma: no cover
            raise ValueError(f"Unsupported tensor library: {self.tensor_library_name}")

        self._bitmasks = [
            self.allocate_token_bitmask(1, self.llg_tokenizer.vocab_size)
            for _ in range(batch_size)
        ]

    def _bias_logits_mlx( # pragma: no cover
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for the MLX backend."""
        import llguidance.mlx
        import llguidance.numpy

        biased_logits_array = []
        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            if not self._is_thinking[i] and not self._generate_end_thinking_token[i]:
                llguidance.numpy.fill_next_token_bitmask(self._ll_matchers[i], self._bitmasks[i], 0)
                biased_logits = llguidance.mlx.apply_token_bitmask(
                    logits[i], self._bitmasks[i] # type: ignore
                )
            elif self._generate_end_thinking_token[i]:
                self._generate_end_thinking_token[i] = False
                biased_logits = llguidance.mlx.apply_token_bitmask(
                    logits[i], self._bitmasks[i] # type: ignore
                )
            else:
                biased_logits = logits[i]

            biased_logits_array.append(biased_logits)

        return self.tensor_adapter.concatenate(biased_logits_array)

    def _bias_logits_torch(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for the Torch backend."""
        import llguidance.torch

        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            if not self._is_thinking[i] and not self._generate_end_thinking_token[i]:
                llguidance.torch.fill_next_token_bitmask(self._ll_matchers[i], self._bitmasks[i], 0)
                llguidance.torch.apply_token_bitmask_inplace(
                    logits[i], self._bitmasks[i] # type: ignore
                )
            elif self._generate_end_thinking_token[i]:
                self._generate_end_thinking_token[i] = False
                llguidance.torch.apply_token_bitmask_inplace(
                    logits[i], self._bitmasks[i] # type: ignore
                )

        return logits

    def _bias_logits_numpy(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for the Numpy backend."""
        import llguidance.numpy

        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            if not self._is_thinking[i] and not self._generate_end_thinking_token[i]:
                llguidance.numpy.fill_next_token_bitmask(self._ll_matchers[i], self._bitmasks[i], 0)
                llguidance.numpy.apply_token_bitmask_inplace(
                    logits[i], self._bitmasks[i] # type: ignore
                )
            elif self._generate_end_thinking_token[i]:
                self._generate_end_thinking_token[i] = False
                llguidance.numpy.apply_token_bitmask_inplace(
                    logits[i], self._bitmasks[i] # type: ignore
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
        batch_size = self.tensor_adapter.shape(input_ids)[0]
        vocab_size = self.tensor_adapter.shape(logits)[1]

        if self.is_first_token:
            self._setup(batch_size)
            self.is_first_token = False
        else:
            self._num_tokens_generated += 1
            for i in range(batch_size):
                latest_token_id = self.tensor_adapter.to_scalar(input_ids[i][-1]) # type: ignore
                if not self._is_thinking[i]:
                    self._ll_matchers[i].consume_token(latest_token_id)
                    error = self._ll_matchers[i].get_error()
                    if error:
                        warnings.warn(f"Error in LLMatcher: {error}")
                else:
                    # If the end of thinking token was generated at the
                    # previous step, we set thinking to False to start
                    # biasing the logits according to the guide
                    if latest_token_id == self.end_thinking_token_id:
                        self._is_thinking[i] = False
                    # If the max number of tokens has been generated, we
                    # modify the bitmask to only allow the end of thinking
                    # token to be generated and set generate_end_thinking_token
                    # to True to skip filling the bitmask (as we did it
                    # manually ourselves)
                    elif (
                        self._num_tokens_generated >= self.thinking_max_tokens
                    ):
                        updated_bitmask = self.tensor_adapter.create_end_thinking_bitmask(
                            vocab_size,
                            self.end_thinking_token_id,
                        )
                        self._bitmasks[i] = self.tensor_adapter.unsqueeze(
                            updated_bitmask # type: ignore
                        )
                        self._generate_end_thinking_token[i] = True

        return self._bias_logits(input_ids, logits)


class LLGuidanceBackend(BaseBackend):
    """Backend for LLGuidance."""

    def __init__(
        self,
        model: SteerableModel,
        *,
        end_thinking_tag: str | None,
        thinking_max_tokens: int | None,
    ):
        """
        Parameters
        ----------
        model
            The Outlines model of the user.
        end_thinking_tag
            The tag the model uses to indicate the end of the thinking process.
            Only used when running a thinking model.
        thinking_max_tokens
            The maximum number of tokens the model can think about. Only used
            when running a thinking model. The end_thinking_tag argument must
            also be provided to use this parameter.

        """
        import llguidance as llg

        self.llg = llg
        self.tensor_library_name = model.tensor_library_name
        self.llg_tokenizer = self._create_llg_tokenizer(model)
        encoded_end_thinking_tag = (
            self.llg_tokenizer.tokenize_str(end_thinking_tag)
            if end_thinking_tag
            else None
        )
        if (
            encoded_end_thinking_tag is not None
            and len(encoded_end_thinking_tag) != 1
        ):
            raise ValueError(
                "The end_thinking_tag must correspond to a single token in"
                + "the tokenizer vocabulary."
            )
        self.end_thinking_token_id = (
            encoded_end_thinking_tag[0] if encoded_end_thinking_tag else None
        )
        self.thinking_max_tokens = thinking_max_tokens

    def _create_llg_tokenizer(self, model: SteerableModel) -> "LLGTokenizer":
        """Create an llg tokenizer from the Outlines model's tokenizer.

        Parameters
        ----------
        model: Model
            The Outlines model.

        Returns
        -------
        LLGTokenizer
            The llg tokenizer.

        """
        if isinstance(model, Transformers):
            import llguidance.hf

            return llguidance.hf.from_tokenizer(model.hf_tokenizer)

        elif isinstance(model, LlamaCpp):
            import llama_cpp
            import llguidance.llamacpp

            vocab = llama_cpp.llama_model_get_vocab(model.model.model)
            return llguidance.llamacpp.lltokenizer_from_vocab(vocab)

        elif isinstance(model, MLXLM): # pragma: no cover
            import llguidance.hf

            return llguidance.hf.from_tokenizer(
                model.mlx_tokenizer._tokenizer
            )

        else: # pragma: no cover
            raise ValueError(
                f"Unsupported model type: {type(model)}. "
                "Llguidance only supports LlamaCpp, MLXLM "
                "and Transformers models."
            )

    def get_json_schema_logits_processor(
        self, json_schema: str
    ) -> LLGuidanceLogitsProcessor:
        """Create a logits processor from a JSON schema.

        Parameters
        ----------
        json_schema: str
            The JSON schema to create a logits processor from.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        grammar_spec = self.llg.grammar_from("json_schema", json_schema)
        return LLGuidanceLogitsProcessor(
            grammar_spec,
            self.llg_tokenizer,
            self.tensor_library_name,
            end_thinking_token_id=self.end_thinking_token_id,
            thinking_max_tokens=self.thinking_max_tokens
        )

    def get_regex_logits_processor(
        self, regex: str
    ) -> LLGuidanceLogitsProcessor:
        """Create a logits processor from a regex.

        Parameters
        ----------
        regex: str
            The regex to create a logits processor from.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        grammar_spec = self.llg.grammar_from("regex", regex)
        return LLGuidanceLogitsProcessor(
            grammar_spec,
            self.llg_tokenizer,
            self.tensor_library_name,
            end_thinking_token_id=self.end_thinking_token_id,
            thinking_max_tokens=self.thinking_max_tokens
        )

    def get_cfg_logits_processor(
        self, grammar: str
    ) -> LLGuidanceLogitsProcessor:
        """Create a logits processor from a context-free grammar.

        Parameters
        ----------
        grammar: str
            The context-free grammar to create a logits processor from.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        # We try both lark and ebnf
        try:
            grammar_spec = self.llg.grammar_from("grammar", grammar)
        except ValueError:
            grammar_spec = self.llg.grammar_from("lark", grammar)
        return LLGuidanceLogitsProcessor(
            grammar_spec,
            self.llg_tokenizer,
            self.tensor_library_name,
            end_thinking_token_id=self.end_thinking_token_id,
            thinking_max_tokens=self.thinking_max_tokens
        )
