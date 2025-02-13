from dataclasses import dataclass
from typing import Any, Optional, Union, cast, get_args

from outlines.models import APIModel, LlamaCpp, LocalModel
from outlines.processors import CFGLogitsProcessor, RegexLogitsProcessor
from outlines.types import CFG, Choice, Json, List, Regex

from .api import SequenceGenerator
from .cfg import cfg
from .choice import choice
from .format import format
from .fsm import fsm
from .json import json
from .regex import regex
from .text import text


@dataclass
class APIGenerator:
    """Represents an API-based generator.

    Attributes
    ----------
    model
        An instance of a model wrapper.
    output_type
        The output type.

    """

    model: APIModel
    output_type: Optional[Union[Json, List, Choice, Regex]] = None

    def __post_init__(self):
        if isinstance(self.output_type, CFG):
            raise NotImplementedError(
                "CFG generation is not supported for API-based models"
            )

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.output_type, **inference_kwargs)


@dataclass
class LocalGenerator:
    """Represents a local model-based generator.

    We use this class to keep track of the logits processor which can be quite
    expensive to build.

    Attributes
    ----------
    model
        An instance of a model wrapper.
    output_type
        The output type.

    """

    model: LocalModel
    output_type: Optional[Union[Json, List, Choice, Regex]]

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
    model: Union[LocalModel, APIModel],
    output_type: Optional[Union[Json, List, Choice, Regex, CFG]] = None,
):
    if isinstance(model, APIModel):  # type: ignore
        return APIGenerator(model, output_type)  # type: ignore
    else:
        return LocalGenerator(model, output_type)  # type: ignore
