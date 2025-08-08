"""Backend class for XGrammar."""

from outlines.backends.base import BaseBackend
from outlines.models import SteerableModel
from outlines.models.mlxlm import MLXLM
from outlines.models.transformers import Transformers
from outlines.processors.base_logits_processor import (
    OutlinesLogitsProcessor,
    TensorType
)


class XGrammarLogitsProcessor(OutlinesLogitsProcessor):
    """Logits processor for XGrammar."""

    def __init__(
        self,
        compiled_grammar: str,
        tensor_library_name: str,
        *,
        end_thinking_token_id: int | None,
        thinking_max_tokens: int | None,
    ):
        """
        Parameters
        ----------
        compiled_grammar: str
            The compiled grammar to use to create the logits processor.
        tensor_library_name: str
            The name of the tensor library used by the model
        end_thinking_token_id: int | None
            The id of the end thinking token
        thinking_max_tokens: int | None
            The maximum number of tokens the model can think about

        """
        import xgrammar as xgr

        self.xgr = xgr
        self.is_first_token = True
        self.compiled_grammar = compiled_grammar
        self.tensor_library_name = tensor_library_name
        self.end_thinking_token_id = end_thinking_token_id
        self.thinking_max_tokens = thinking_max_tokens or float("inf")
        super().__init__(tensor_library_name)

    def reset(self):
        """Ensure self._setup is called again for the next generation."""
        self.is_first_token = True

    def _setup(self, batch_size: int, vocab_size: int) -> None:
        """Setup the logits processor for a new generation."""
        self._matchers = [
            self.xgr.GrammarMatcher(self.compiled_grammar)
            for _ in range(batch_size)
        ]
        self._bitmasks = [
            self.xgr.allocate_token_bitmask(1, vocab_size)
            for _ in range(batch_size)
        ]
        self._is_thinking = [self.end_thinking_token_id is not None] * batch_size
        self._generate_end_thinking_token = [False] * batch_size
        self._num_tokens_generated = 0

        if self.tensor_library_name == "torch":
            self._bias_logits = self._bias_logits_torch
        elif self.tensor_library_name == "mlx":
            self._bias_logits = self._bias_logits_mlx
        else: # pragma: no cover
            raise ValueError(
                f"Unsupported tensor library: {self.tensor_library_name}"
            )

    def _bias_logits_torch(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for Torch tensors."""
        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            if not self._is_thinking[i] and not self._generate_end_thinking_token[i]:
                if not self._matchers[i].is_terminated():
                    self._matchers[i].fill_next_token_bitmask(self._bitmasks[i], 0)
                self.xgr.apply_token_bitmask_inplace(logits[i], self._bitmasks[i])
            elif self._generate_end_thinking_token[i]:
                self._generate_end_thinking_token[i] = False
                self.xgr.apply_token_bitmask_inplace(logits[i], self._bitmasks[i])

        return logits

    def _bias_logits_mlx( # pragma: no cover
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for MLX tensors."""
        import mlx.core as mx
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        biased_logits_array = []

        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            if not self._is_thinking[i] and not self._generate_end_thinking_token[i]:
                if not self._matchers[i].is_terminated():
                    self._matchers[i].fill_next_token_bitmask(self._bitmasks[i], 0)
                biased_logits = apply_token_bitmask_mlx(
                    mx.array(self._bitmasks[i].numpy()), logits[i], 1
                )
            elif self._generate_end_thinking_token[i]:
                self._generate_end_thinking_token[i] = False
                biased_logits = apply_token_bitmask_mlx(
                    mx.array(self._bitmasks[i].numpy()), logits[i], 1
                )
            else:
                biased_logits = logits[i]

            biased_logits_array.append(biased_logits)

        return self.tensor_adapter.concatenate(biased_logits_array)

    def process_logits(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Use the XGrammar matchers to bias the logits."""
        batch_size = self.tensor_adapter.shape(input_ids)[0]
        vocab_size = self.tensor_adapter.shape(logits)[1]

        if self.is_first_token:
            self._setup(batch_size, vocab_size)
            self.is_first_token = False
        else:
            self._num_tokens_generated += 1
            for i in range(batch_size):
                latest_token_id = self.tensor_adapter.to_scalar(
                    input_ids[i][-1] # type: ignore
                )
                if not self._is_thinking[i]:
                    if not self._matchers[i].is_terminated():
                        assert self._matchers[i].accept_token(latest_token_id)
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


class XGrammarBackend(BaseBackend):
    """Backend for XGrammar."""

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
        import xgrammar as xgr

        if isinstance(model, Transformers):
            tokenizer = model.hf_tokenizer
        elif isinstance(model, MLXLM):
            tokenizer = model.mlx_tokenizer._tokenizer
        else: # pragma: no cover
            raise ValueError(
                "The xgrammar backend only supports Transformers and "
                + "MLXLM models"
            )

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=len(tokenizer.get_vocab())
        )
        self.grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        self.tensor_library_name = model.tensor_library_name
        encoded_end_thinking_tag = (
            tokenizer.encode(end_thinking_tag)
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
            encoded_end_thinking_tag[0]
            if encoded_end_thinking_tag is not None
            else None
        )
        self.thinking_max_tokens = thinking_max_tokens

    def get_json_schema_logits_processor(
        self, json_schema: str
    ) -> XGrammarLogitsProcessor:
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
        compiled_grammar = self.grammar_compiler.compile_json_schema(
            json_schema
        )
        return XGrammarLogitsProcessor(
            compiled_grammar,
            self.tensor_library_name,
            end_thinking_token_id=self.end_thinking_token_id,
            thinking_max_tokens=self.thinking_max_tokens
        )

    def get_regex_logits_processor(
        self, regex: str
    ) -> XGrammarLogitsProcessor:
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
        compiled_grammar = self.grammar_compiler.compile_regex(regex)
        return XGrammarLogitsProcessor(
            compiled_grammar,
            self.tensor_library_name,
            end_thinking_token_id=self.end_thinking_token_id,
            thinking_max_tokens=self.thinking_max_tokens
        )

    def get_cfg_logits_processor(
        self, grammar: str
    ) -> XGrammarLogitsProcessor:
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
        compiled_grammar = self.grammar_compiler.compile_grammar(grammar)
        return XGrammarLogitsProcessor(
            compiled_grammar,
            self.tensor_library_name,
            end_thinking_token_id=self.end_thinking_token_id,
            thinking_max_tokens=self.thinking_max_tokens
        )
