from dataclasses import dataclass
from typing import Any, Optional, Union, cast, get_args

import interegular

from outlines.fsm.guide import RegexGuide
from outlines.models import BlackBoxModel, SteerableModel
from outlines.processors import CFGLogitsProcessor, GuideLogitsProcessor, RegexLogitsProcessor
from outlines.types import CFG, Choice, JsonType, List, Regex


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
    output_type: Optional[Union[JsonType, List, Choice, Regex]] = None

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
    output_type: Optional[Union[JsonType, List, Choice, Regex, CFG, interegular.fsm.FSM]]

    def __post_init__(self):
        if self.output_type is None:
            self.logits_processor = None
        else:
            if isinstance(self.output_type, CFG):
                cfg_string = self.output_type.definition
                self.logits_processor = CFGLogitsProcessor(
                    cfg_string, self.model.tokenizer
                )
            elif isinstance(self.output_type, interegular.fsm.FSM):
                guide = RegexGuide.from_interegular_fsm(self.output_type, self.model.tokenizer)
                self.logits_processor = GuideLogitsProcessor(tokenizer=self.model.tokenizer, guide=guide)
            else:
                regex_string = self.output_type.to_regex()
                self.logits_processor = RegexLogitsProcessor(
                    regex_string, self.model.tokenizer
                )

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.logits_processor, **inference_kwargs)


def Generator(
    model: Union[SteerableModel, BlackBoxModel],
    output_type: Optional[Union[JsonType, List, Choice, Regex, CFG, interegular.fsm.FSM]] = None,
):
    if isinstance(model, BlackBoxModel):  # type: ignore
        return BlackBoxGenerator(model, output_type)  # type: ignore
    else:
        return SteerableGenerator(model, output_type)  # type: ignore
