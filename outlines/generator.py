"""Encapsulate a model and an output type into a reusable object."""

from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Union,
)

from outlines.models import (
    AsyncBlackBoxModel,
    BlackBoxModel,
    SteerableModel,
)
from outlines.models.base import AsyncModel, Model
from outlines.processors import (
    CFGLogitsProcessor,
    GuideLogitsProcessor,
    OutlinesLogitsProcessor,
    RegexLogitsProcessor,
)
from outlines.processors.guide import RegexGuide
from outlines.types import CFG, FSM
from outlines.types.dsl import python_types_to_terms, to_regex


class BlackBoxGenerator:
    """Synchronous generator for which we don't control constrained
    generation.

    The output type provided is not compiled into a logits processor, but is
    instead directly passed on to the model.

    """
    output_type: Optional[Any]

    def __init__(self, model: BlackBoxModel, output_type: Optional[Any]):
        """
        Parameters
        ----------
        model
            An instance of an Outlines model.
        output_type
            The output type that will be used to constrain the generation.

        """
        self.model = model
        self.output_type = output_type

        if isinstance(self.output_type, FSM):
            raise NotImplementedError(
                "FSM generation is not supported for API-based models"
            )

    def __call__(self, prompt: Any, **inference_kwargs) -> Any:
        """Generate a response from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        return self.model.generate(
            prompt, self.output_type, **inference_kwargs
        )

    def batch(self, prompts: List[Any], **inference_kwargs) -> List[Any]:
        """Generate a batch of responses from the model.

        Parameters
        ----------
        prompts
            The list of prompts to use to generate a batch of responses.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        List[Any]
            The list of responses generated by the model.

        """
        return self.model.generate_batch(
            prompts, self.output_type, **inference_kwargs
        )

    def stream(self, prompt: Any, **inference_kwargs) -> Iterator[Any]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        return self.model.generate_stream(
            prompt, self.output_type, **inference_kwargs
        )


class AsyncBlackBoxGenerator:
    """Asynchronous generator for which we don't control constrained
    generation.

    The output type provided is not compiled into a logits processor, but is
    instead directly passed on to the model.

    """
    output_type: Optional[Any]

    def __init__(self, model: AsyncBlackBoxModel, output_type: Optional[Any]):
        """
        Parameters
        ----------
        model
            An instance of an Outlines model.
        output_type
            The output type that will be used to constrain the generation.

        """
        self.model = model
        self.output_type = output_type

        if isinstance(self.output_type, FSM):
            raise NotImplementedError(
                "FSM generation is not supported for API-based models"
            )

    async def __call__(self, prompt: Any, **inference_kwargs) -> Any:
        """Generate a response from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        return await self.model.generate(
            prompt, self.output_type, **inference_kwargs
        )

    async def batch(self, prompts: List[Any], **inference_kwargs) -> List[Any]:
        """Generate a batch of responses from the model.

        Parameters
        ----------
        prompts
            The list of prompts to use to generate a batch of responses.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        List[Any]
            The list of responses generated by the model.

        """
        return await self.model.generate_batch(
            prompts, self.output_type, **inference_kwargs
        )

    async def stream(self, prompt: Any, **inference_kwargs) -> AsyncIterator[Any]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        async for chunk in self.model.generate_stream(  # pragma: no cover
            prompt, self.output_type, **inference_kwargs
        ):
            yield chunk


class SteerableGenerator:
    """Represents a generator for which we control constrained generation.

    The generator is responsible for building and storing the logits processor
    (which can be quite expensive to build), and then passing it to the model
    when the generator is called.

    The argument defining constrained generation can be of 2 types associated
    to different methods to create an instance of the generator:
    - `output_type` (through `__init__`): an output type as defined in the
      `outlines.types` module
    - `processor` (through `from_processor`): an already built logits processor
       as defined in the `outlines.processors` module

    The 2 parameters are mutually exclusive.

    """
    logits_processor: Optional[OutlinesLogitsProcessor]

    def __init__(self, model: SteerableModel, output_type: Optional[Any]):
        """
        Parameters
        ----------
        model
            An instance of an Outlines model.
        output_type
            The output type expressed as a Python type

        """
        self.model = model
        if output_type is None:
            self.logits_processor = None
        else:
            term = python_types_to_terms(output_type)
            if isinstance(term, CFG):
                cfg_string = term.definition
                self.logits_processor = CFGLogitsProcessor(
                    cfg_string,
                    self.model.tokenizer,
                    self.model.tensor_library_name,
                )
            elif isinstance(term, FSM):
                guide = RegexGuide.from_interegular_fsm(
                    term.fsm,
                    self.model.tokenizer,
                )
                self.logits_processor = GuideLogitsProcessor(
                    self.model.tokenizer, guide, self.model.tensor_library_name
                )
            else:
                regex_string = to_regex(term)
                self.logits_processor = RegexLogitsProcessor(
                    regex_string,
                    self.model.tokenizer,
                    self.model.tensor_library_name,
                )

    @classmethod
    def from_processor(
        cls, model: SteerableModel, processor: OutlinesLogitsProcessor
    ):
        """Create a generator from a logits processor.

        Parameters
        ----------
        model
            An instance of an Outlines model.
        processor
            An instance of an OutlinesLogitsProcessor.

        """
        if not isinstance(processor, OutlinesLogitsProcessor):
            raise TypeError(
                "The processor argument must be an instance of "
                "OutlinesLogitsProcessor"
            )
        instance = cls.__new__(cls)
        instance.model = model
        instance.logits_processor = processor

        return instance

    def __call__(self, prompt: Any, **inference_kwargs) -> Any:
        """Generate a response from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        return self.model.generate(
            prompt, self.logits_processor, **inference_kwargs
        )

    def batch(self, prompts: List[Any], **inference_kwargs) -> List[Any]:
        """Generate a batch of responses from the model.

        Parameters
        ----------
        prompts
            The list of prompts to use to generate a batch of responses.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        List[Any]
            The list of responses generated by the model.

        """
        return self.model.generate_batch(
            prompts, self.logits_processor, **inference_kwargs
        )

    def stream(self, prompt: Any, **inference_kwargs) -> Iterator[Any]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        return self.model.generate_stream(
            prompt, self.logits_processor, **inference_kwargs
        )


def Generator(
    model: Union[Model, AsyncModel],
    output_type: Optional[Any] = None,
    *,
    processor: Optional[OutlinesLogitsProcessor] = None,
) -> Union[SteerableGenerator, BlackBoxGenerator, AsyncBlackBoxGenerator]:
    """Create a generator for the given model and output parameters.

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
    Union[SteerableGenerator, BlackBoxGenerator, AsyncBlackBoxGenerator]
        A generator instance.

    """
    provided_output_params = sum(
        param is not None
        for param in [output_type, processor]
    )
    if provided_output_params > 1:
        raise ValueError(
            "At most one of output_type or processor can be provided"
        )

    if isinstance(model, SteerableModel): # type: ignore
        if processor is not None:
            return SteerableGenerator.from_processor(model, processor) # type: ignore
        else:
            return SteerableGenerator(model, output_type) # type: ignore
    else:
        if processor is not None:
            raise NotImplementedError(
                "This model does not support logits processors"
            )
        if isinstance(model, AsyncBlackBoxModel): # type: ignore
            return AsyncBlackBoxGenerator(model, output_type) # type: ignore
        elif isinstance(model, BlackBoxModel): # type: ignore
            return BlackBoxGenerator(model, output_type) # type: ignore
        else:
            raise ValueError(
                "The model argument must be an instance of "
                "SteerableModel, BlackBoxModel or AsyncBlackBoxModel"
            )
