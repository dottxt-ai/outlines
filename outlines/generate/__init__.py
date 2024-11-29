from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, cast, get_args

from outlines.models import APIModel, LocalModel
from outlines.types import Choice, Json, List, Regex

from .api import SequenceGenerator
from .cfg import cfg
from .choice import choice
from .format import format
from .fsm import fsm
from .json import json
from .regex import regex
from .text import text

if TYPE_CHECKING:
    from outlines.processors import RegexLogitsProcessor


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
            regex_string = self.output_type.to_regex()
            self.logits_processor = RegexLogitsProcessor(
                regex_string, self.model.tokenizer
            )

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.logits_processor, **inference_kwargs)


def Generator(
    model: Union[LocalModel, APIModel],
    output_type: Optional[Union[Json, List, Choice, Regex]] = None,
):
    if isinstance(model, APIModel):  # type: ignore
        return APIGenerator(model, output_type)  # type: ignore
    else:
        return LocalGenerator(model, output_type)  # type: ignore
