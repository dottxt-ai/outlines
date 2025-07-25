"""Backend class for XGrammar."""

from typing import TYPE_CHECKING

from outlines.backends.base import BaseBackend
from outlines.models import SteerableModel
from outlines.models.transformers import Transformers
from outlines.processors.base_logits_processor import (
    OutlinesLogitsProcessor,
    TensorType
)

if TYPE_CHECKING:
    from xgrammar.contrib.hf import LogitsProcessor


class XGrammarLogitsProcessor(OutlinesLogitsProcessor):
    """Logits processor for XGrammar.

    This class wraps the `xgr.contrib.hf.LogitsProcessor` class and adds
    a `reset` method to reset the logits processor for a new generation.

    """

    def __init__(self, compiled_grammar: str):
        """
        Parameters
        ----------
        compiled_grammar: str
            The compiled grammar to use to create the logits processor.

        """
        import xgrammar as xgr

        self.xgr = xgr
        self.compiled_grammar = compiled_grammar
        self.xgrammar_logits_processor = None
        super().__init__("torch")

    def reset(self):
        """Reset the logits processor for a new generation."""
        self.xgrammar_logits_processor = None

    def process_logits(self, input_ids: TensorType, logits: TensorType) -> TensorType:
        """Bias the logits."""
        if self.xgrammar_logits_processor is None:
            self.xgrammar_logits_processor = self.xgr.contrib.hf.LogitsProcessor(
                self.compiled_grammar
            )
        return self.xgrammar_logits_processor(input_ids, logits) # type: ignore


class XGrammarBackend(BaseBackend):
    """Backend for XGRammar."""

    def __init__(self, model: SteerableModel):
        """
        Parameters
        ----------
        model
            The Outlines model of the user.

        """
        import xgrammar as xgr
        from transformers import AutoConfig

        if not isinstance(model, Transformers):
            raise ValueError(
                "The xgrammar backend only supports Transformers models"
            )

        vocab_size = AutoConfig.from_pretrained(
            model.model.config._name_or_path
        ).vocab_size
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            model.hf_tokenizer,
            vocab_size=vocab_size
        )
        self.grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

    def get_json_schema_logits_processor(
        self, json_schema: str
    ) -> "LogitsProcessor":
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
        return XGrammarLogitsProcessor(compiled_grammar)

    def get_regex_logits_processor(self, regex: str) -> "LogitsProcessor":
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
        return XGrammarLogitsProcessor(compiled_grammar)

    def get_cfg_logits_processor(self, grammar: str) -> "LogitsProcessor":
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
        return XGrammarLogitsProcessor(compiled_grammar)

    def get_fsm_logits_processor(self, fsm):
        raise NotImplementedError(
            "XGrammarBackend does not support FSM logits processors. "
            "Use the outlines_core backend instead."
        )
