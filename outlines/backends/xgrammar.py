"""Backend class for XGrammar."""

from typing import TYPE_CHECKING

from outlines.backends.base import BaseBackend
from outlines.models import SteerableModel
from outlines.models.transformers import Transformers

if TYPE_CHECKING:
    from xgrammar.contrib.hf import LogitsProcessor


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

        self.xgr = xgr
        vocab_size = AutoConfig.from_pretrained(
            model.model.config._name_or_path
        ).vocab_size
        tokenizer_info = self.xgr.TokenizerInfo.from_huggingface(
            model.hf_tokenizer,
            vocab_size=vocab_size
        )
        self.grammar_compiler = self.xgr.GrammarCompiler(tokenizer_info)

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
        xgr_logits_processor = self.xgr.contrib.hf.LogitsProcessor(
            compiled_grammar
        )
        return xgr_logits_processor

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
        xgr_logits_processor = self.xgr.contrib.hf.LogitsProcessor(
            compiled_grammar
        )
        return xgr_logits_processor

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
        xgr_logits_processor = self.xgr.contrib.hf.LogitsProcessor(
            compiled_grammar
        )
        return xgr_logits_processor

    def get_fsm_logits_processor(self, fsm):
        raise NotImplementedError(
            "XGrammarBackend does not support FSM logits processors. "
            "Use the outlines_core backend instead."
        )
