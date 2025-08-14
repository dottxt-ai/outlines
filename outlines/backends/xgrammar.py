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

    def __init__(self, compiled_grammar: str, tensor_library_name: str,):
        """
        Parameters
        ----------
        compiled_grammar: str
            The compiled grammar to use to create the logits processor.
        tensor_library_name: str
            The name of the tensor library used by the model

        """
        import xgrammar as xgr

        self.xgr = xgr
        self.is_first_token = True
        self.compiled_grammar = compiled_grammar
        self.tensor_library_name = tensor_library_name
        super().__init__(tensor_library_name)

    def reset(self):
        """Ensure self._setup is called again for the next generation."""
        self.is_first_token = True

    def _setup(self, batch_size: int, vocab_size: int) -> None:
        """Setup the logits processor for a new generation."""
        if self.tensor_library_name == "torch":
            self._bias_logits = self._bias_logits_torch
        elif self.tensor_library_name == "mlx": # pragma: no cover
            self._bias_logits = self._bias_logits_mlx
        else: # pragma: no cover
            raise ValueError(
                f"Unsupported tensor library: {self.tensor_library_name}"
            )

        self._matchers = [
            self.xgr.GrammarMatcher(self.compiled_grammar)
            for _ in range(batch_size)
        ]
        self._bitmask = self.xgr.allocate_token_bitmask(batch_size, vocab_size)

    def _bias_logits_torch(
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for Torch tensors."""
        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            if not self._matchers[i].is_terminated():
                self._matchers[i].fill_next_token_bitmask(self._bitmask, i)

        self._bitmask = self.tensor_adapter.to_device(
            self._bitmask,
            self.tensor_adapter.get_device(logits)
        )
        self.xgr.apply_token_bitmask_inplace(logits, self._bitmask)
        self._bitmask = self.tensor_adapter.to_device(
            self._bitmask,
            "cpu"
        )

        return logits

    def _bias_logits_mlx( # pragma: no cover
        self, input_ids: TensorType, logits: TensorType
    ) -> TensorType:
        """Bias the logits for MLX tensors."""
        import mlx.core as mx
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        for i in range(self.tensor_adapter.shape(input_ids)[0]):
            if not self._matchers[i].is_terminated():
                self._matchers[i].fill_next_token_bitmask(self._bitmask, i)

        biased_logits = apply_token_bitmask_mlx(
            mx.array(self._bitmask.numpy()), logits, self.tensor_adapter.shape(logits)[1]
        )

        return biased_logits

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
            for i in range(batch_size):
                if not self._matchers[i].is_terminated(): # pragma: no cover
                    last_token_id = self.tensor_adapter.to_scalar(
                        input_ids[i][-1] # type: ignore
                    )
                    assert self._matchers[i].accept_token(last_token_id)

        return self._bias_logits(input_ids, logits)


class XGrammarBackend(BaseBackend):
    """Backend for XGrammar."""

    def __init__(self, model: SteerableModel):
        """
        Parameters
        ----------
        model
            The Outlines model of the user.

        """
        import xgrammar as xgr

        if isinstance(model, Transformers):
            tokenizer = model.hf_tokenizer
        elif isinstance(model, MLXLM): # pragma: no cover
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
            self.tensor_library_name
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
            self.tensor_library_name
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
            self.tensor_library_name
        )
