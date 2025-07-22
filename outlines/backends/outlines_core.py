"""Backend class for Outlines Core."""

from interegular.fsm import FSM

from outlines.backends.base import BaseBackend
from outlines.models import SteerableModel
from outlines.processors import (
    CFGLogitsProcessor,
    GuideLogitsProcessor,
    RegexLogitsProcessor,
)
from outlines.processors.guide import RegexGuide
from outlines_core.fsm.json_schema import build_regex_from_schema


class OutlinesCoreBackend(BaseBackend):
    """Backend for Outlines Core."""

    def __init__(self, model: SteerableModel):
        """
        Parameters
        ----------
        model
            The Outlines model of the user.

        """
        self.tokenizer = model.tokenizer
        self.tensor_library_name = model.tensor_library_name

    def get_json_schema_logits_processor(
        self, json_schema: str
    ) -> RegexLogitsProcessor:
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
        regex = build_regex_from_schema(json_schema)
        return self.get_regex_logits_processor(regex)

    def get_regex_logits_processor(self, regex: str) -> RegexLogitsProcessor:
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
        return RegexLogitsProcessor(
            regex,
            self.tokenizer,
            self.tensor_library_name,
        )

    def get_cfg_logits_processor(self, grammar: str) -> CFGLogitsProcessor:
        """Create a logits processor from a context-free grammar.

        Parameters
        ----------
        grammar: str
            The context-free grammar to create a logits processor from. For
            Outlines Core, the grammar must use the Lark format.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        return CFGLogitsProcessor(
            grammar,
            self.tokenizer,
            self.tensor_library_name,
        )

    def get_fsm_logits_processor(self, fsm: FSM) -> GuideLogitsProcessor:
        """Create a logits processor from an interegular FSM.

        Parameters
        ----------
        fsm: interegular.fsm.FSM
            The interegular FSM to create a logits processor from.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        guide = RegexGuide.from_interegular_fsm(fsm, self.tokenizer)
        return GuideLogitsProcessor(
            self.tokenizer,
            guide,
            self.tensor_library_name,
        )
