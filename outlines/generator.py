from typing import Any, Optional, Union

from outlines.models import BlackBoxModel, SteerableModel
from outlines.processors import (
    CFGLogitsProcessor,
    GuideLogitsProcessor,
    RegexLogitsProcessor,
    OutlinesLogitsProcessor,
)
from outlines.processors.guide import RegexGuide
from outlines.types import CFG, FSM
from outlines.types.dsl import python_types_to_terms, to_regex


class BlackBoxGenerator:
    """Represents a generator for which we don't control constrained generation.

    This type of generator only accepts an output type as an argument defining
    constrained generation. This output type is not modified and thus only
    passed through to the model.
    """
    output_type: Optional[Any]

    def __init__(self, model, output_type: Optional[Any]):
        self.model = model
        self.output_type = output_type

        if isinstance(self.output_type, CFG):
            raise NotImplementedError(
                "CFG generation is not supported for API-based models"
            )
        elif isinstance(self.output_type, FSM):
            raise NotImplementedError(
                "FSM generation is not supported for API-based models"
            )

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.output_type, **inference_kwargs)

    def stream(self, prompt, **inference_kwargs):
        return self.model.generate_stream(prompt, self.output_type, **inference_kwargs)


class SteerableGenerator:
    """Represents a generator for which we control constrained generation.

    The generator is responsible for building and storing the logits processor
    (which can be quite expensive to build), and then to pass it to the model
    when the generator is called. The argument defining constrained generation
    can be of 2 types associated to different methods to create an instance of
    the generator:
    - output_type (through `__init__`): a Python type, will be used to build a
      logits processor
    - processor (through `from_processor`): an already built
      logits processor
    """
    logits_processor: Optional[OutlinesLogitsProcessor]

    def __init__(self, model, output_type: Optional[Any]):
        self.model = model

        if output_type is None:
            self.logits_processor = None
        else:
            term = python_types_to_terms(output_type)
            if isinstance(term, CFG):
                cfg_string = term.definition
                self.logits_processor = CFGLogitsProcessor(
                    cfg_string, self.model.tokenizer
                )
            elif isinstance(term, FSM):
                guide = RegexGuide.from_interegular_fsm(term.fsm, self.model.tokenizer)
                self.logits_processor = GuideLogitsProcessor(tokenizer=self.model.tokenizer, guide=guide)
            else:
                regex_string = to_regex(term)
                self.logits_processor = RegexLogitsProcessor(
                    regex_string, self.model.tokenizer
                )

    @classmethod
    def from_processor(cls, model, processor: OutlinesLogitsProcessor):
        if not isinstance(processor, OutlinesLogitsProcessor):
            raise TypeError("The processor argument must be an instance of OutlinesLogitsProcessor")
        instance = cls.__new__(cls)
        instance.model = model
        instance.logits_processor = processor

        return instance

    def __call__(self, prompt, **inference_kwargs):
        return self.model.generate(prompt, self.logits_processor, **inference_kwargs)

    def stream(self, prompt, **inference_kwargs):
        return self.model.generate_stream(prompt, self.logits_processor, **inference_kwargs)


def Generator(
    model: Union[SteerableModel, BlackBoxModel],
    output_type: Optional[Any] = None,
    *,
    processor: Optional[OutlinesLogitsProcessor] = None,
):
    """
    Create a generator for the given model and output parameters.

    The 2 parameters output_type and processor are mutually exclusive. The
    parameters processor is only supported for SteerableModel instances
    (typically local models) and is intended to be only used by advanced users.

    Parameters
    ----------
    model
        An instance of an Outlines model.
    output_type
        The output type expressed as a Python type or a type defined in the
        outlines.types.dsl module.
    processor
        An instance of an OutlinesLogitsProcessor.

    Returns
    -------
    A generator.
    """
    provided_output_params = sum(
        param is not None
        for param in [output_type, processor]
    )
    if provided_output_params > 1:
        raise ValueError("At most one of output_type or processor can be provided")

    if isinstance(model, BlackBoxModel): # type: ignore
        if processor is not None:
            raise NotImplementedError("This model does not support logits processors")
        else:
            return BlackBoxGenerator(model, output_type)
    else:
        if processor is not None:
            return SteerableGenerator.from_processor(model, processor)
        else:
            return SteerableGenerator(model, output_type)
