from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, get_args

from outlines.models import APIModel, Model
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
class Generator:
    """Represents a generator.

    We use a class that can hold the logits processor which can be quite
    expensive to build.

    Attributes
    ----------
    model
        An instance of a model wrapper.
    output_description
        The output type for API models, and the processor for other models.

    """

    model: Union[APIModel, Model]
    output_description: Union["RegexLogitsProcessor", Any]

    def __init__(
        self, model, output_type: Optional[Union[Json, List, Choice, Regex]] = None
    ):
        from outlines.processors import RegexLogitsProcessor

        if isinstance(model, get_args(APIModel)):
            self.output_description = output_type
        else:
            if output_type is not None:
                regex_string = output_type.to_regex()
                self.output_description = RegexLogitsProcessor(
                    regex_string, model.tokenizer
                )
            else:
                self.output_description = None

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.output_description, **inference_kwargs)
