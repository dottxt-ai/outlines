from dataclasses import dataclass
from typing import Any, Optional, Union, cast, get_args

from outlines.models import BlackBoxModel, SteerableModel
from outlines.processors import CFGLogitsProcessor, RegexLogitsProcessor
from outlines.types import CFG, Choice, JsonType, List, Regex

from .api import SequenceGenerator
from .cfg import cfg
from .choice import choice
from .format import format
from .fsm import fsm
from .json import json
from .regex import regex
from .text import text


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
    output_type: Optional[Union[JsonType, List, Choice, Regex]]

    def __post_init__(self):
        if self.output_type is None:
            self.logits_processor = None
        else:
            if isinstance(self.output_type, CFG):
                cfg_string = self.output_type.definition
                self.logits_processor = CFGLogitsProcessor(
                    cfg_string, self.model.tokenizer
                )
            else:
                regex_string = self.output_type.to_regex()
                self.logits_processor = RegexLogitsProcessor(
                    regex_string, self.model.tokenizer
                )

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.logits_processor, **inference_kwargs)


def Generator(
    model: Union[SteerableModel, BlackBoxModel],
    output_type: Optional[Union[JsonType, List, Choice, Regex, CFG]] = None,
):
    if isinstance(model, BlackBoxModel):  # type: ignore
        return BlackBoxGenerator(model, output_type)  # type: ignore
    else:
        return SteerableGenerator(model, output_type)  # type: ignore
