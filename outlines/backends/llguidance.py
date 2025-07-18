"""Backend class for LLGuidance."""

from typing import TYPE_CHECKING

from outlines.backends.base import BaseBackend
from outlines.models import LlamaCpp, MLXLM, SteerableModel, Transformers
from outlines.processors.llguidance import LLGuidanceLogitsProcessor

if TYPE_CHECKING:
    from llguidance import LLGTokenizer


class LLGuidanceBackend(BaseBackend):
    """Backend for LLGuidance."""

    def __init__(self, model: SteerableModel):
        """
        Parameters
        ----------
        model
            The Outlines model of the user.

        """
        import llguidance as llg

        self.llg = llg
        self.tensor_library_name = model.tensor_library_name
        self.llg_tokenizer = self._create_llg_tokenizer(model)

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

        elif isinstance(model, MLXLM):
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
            grammar_spec, self.llg_tokenizer, self.tensor_library_name
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
            grammar_spec, self.llg_tokenizer, self.tensor_library_name
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
        grammar_spec = self.llg.grammar_from("lark", grammar)
        return LLGuidanceLogitsProcessor(
            grammar_spec, self.llg_tokenizer, self.tensor_library_name
        )

    def get_fsm_logits_processor(self, fsm):
        raise NotImplementedError(
            "LLGuidanceBackend does not support FSM logits processors. "
            "Use the outlines_core backend instead."
        )
