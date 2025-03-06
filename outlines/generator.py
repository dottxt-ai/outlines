from dataclasses import dataclass
from typing import Any, Optional, Union, cast, get_args

import interegular

from outlines.fsm.guide import RegexGuide
from outlines.models import BlackBoxModel, SteerableModel
from outlines.processors import CFGLogitsProcessor, GuideLogitsProcessor, RegexLogitsProcessor
from outlines.types import CFG
from outlines.types.dsl import python_types_to_terms, to_regex


@dataclass
class BlackBoxGenerator:
    """Represents a generator for which we don't control constrained generation.

    Attributes
    ----------
    model
        An instance of a model wrapper.
    output_type
        The output type.

    """

    model: BlackBoxModel
    output_type: Optional[Any] = None

    def __post_init__(self):
        if isinstance(self.output_type, CFG):
            raise NotImplementedError(
                "CFG generation is not supported for API-based models"
            )
        elif isinstance(self.output_type, interegular.fsm.FSM):
            raise NotImplementedError(
                "FSM generation is not supported for API-based models"
            )

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.output_type, **inference_kwargs)


@dataclass
class SteerableGenerator:
    """Represents a generator for which we control constrained generation.

    We use this class to keep track of the logits processor which can be quite
    expensive to build.

    Attributes
    ----------
    model
        An instance of a model wrapper.
    output_type
        The output type.

    """

    model: SteerableModel
    output_type: Optional[Any]

    def __post_init__(self):
        # If CFG -> CFG
        # if dict -> CFG
        # if Any -> None
        # Else -> Regex
        if self.output_type is None:
            self.logits_processor = None
        else:
            term = python_types_to_terms(self.output_type)
            if isinstance(term, CFG):
                cfg_string = term.definition
                self.logits_processor = CFGLogitsProcessor(
                    cfg_string, self.model.tokenizer
                )
            elif isinstance(term, interegular.fsm.FSM):
                guide = RegexGuide.from_interegular_fsm(term, self.model.tokenizer)
                self.logits_processor = GuideLogitsProcessor(tokenizer=self.model.tokenizer, guide=guide)
            else:
                regex_string = to_regex(term)
                self.logits_processor = RegexLogitsProcessor(
                    regex_string, self.model.tokenizer
                )

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.logits_processor, **inference_kwargs)


def Generator(
    model: Union[SteerableModel, BlackBoxModel],
    output_type: Optional[Any] = None,
):
    if isinstance(model, BlackBoxModel):  # type: ignore
        return BlackBoxGenerator(model, output_type)  # type: ignore
    else:
        return SteerableGenerator(model, output_type)  # type: ignore
