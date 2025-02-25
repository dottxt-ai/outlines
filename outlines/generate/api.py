import datetime
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

if TYPE_CHECKING:
    import torch

FormattedOutput = Union[
    str, int, float, bool, datetime.date, datetime.time, datetime.datetime
]


@dataclass(frozen=True)
class GenerationParameters:
    """Generation parameters used in Outlines' public API."""

    max_tokens: Optional[int]
    stop_at: Optional[Union[str, List[str]]]
    seed: Optional[int]


@dataclass(frozen=True)
class SamplingParameters:
    """Sampling parameters available in Outlines."""

    sampler: str
    num_samples: int = 1
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


class SequenceGeneratorAdapter:
    """Class used to unify the interface to the model providers'
    generation functions.

    Attributes
    ----------
    model
        The wrapped model.
    logits_processor
        The logits processor to use to generate text.
    sampler
        The sampler to use to generate text.

    """

    def __init__(self, model, logits_processor, sampler):
        self.model = model
        self.logits_processor = logits_processor

        self.sampling_params = sampler.sampling_params

    def prepare_generation_parameters(
        self,
        max_tokens: Optional[int],
        stop_at: Optional[Union[str, List[str]]],
        seed: Optional[int],
    ):
        if isinstance(stop_at, str):
            stop_at = [stop_at]

        generation_params = GenerationParameters(
            max_tokens,
            stop_at,
            seed,
        )

        return generation_params

    def format_sequence(self, sequence: str) -> FormattedOutput:
        """Translate the generated sequence to another type.

        This method is for instance overridden when generating JSON to either
        return a dictionnary or a Pydantic model.

        Parameters
        ----------
        sequence
            A generated sequences.

        Returns
        -------
        The formatted sequence.

        """
        return sequence

    def _format(self, sequences):
        """Apply formatting to every string in a completion."""
        if isinstance(sequences, list):
            return [self._format(sequence) for sequence in sequences]
        else:
            return self.format_sequence(sequences)

    def __call__(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Generate text from a prompt of list of prompts."""

        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )

        completions = self.model.generate(
            prompts,
            generation_params,
            copy(self.logits_processor),
            self.sampling_params,
            **model_specific_params,
        )

        return self._format(completions)

    def stream(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Return a text generator from a prompt or a list of prompts."""
        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )
        return self.model.stream(
            prompts,
            generation_params,
            copy(self.logits_processor),
            self.sampling_params,
            **model_specific_params,
        )


class VisionSequenceGeneratorAdapter(SequenceGeneratorAdapter):
    def __call__(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[str, Any],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """
        Generate text from a prompt of list of prompts.

        Media: A URI to construct media or media object itself. Used as AutoProcessor argument.
        """
        prompts, media = self._validate_prompt_media_types(prompts, media)

        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )

        completions = self.model.generate(
            prompts,
            media,
            generation_params,
            copy(self.logits_processor),
            self.sampling_params,
            **model_specific_params,
        )

        return self._format(completions)

    def stream(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: List[Union[str, Any, List[Union[str, Any]]]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Return a text generator from a prompt or a list of prompts."""
        prompts, media = self._validate_prompt_media_types(prompts, media)
        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )
        return self.model.stream(
            prompts,
            media,
            generation_params,
            copy(self.logits_processor),
            self.sampling_params,
            **model_specific_params,
        )

    @classmethod
    def _validate_prompt_media_types(
        cls,
        prompts: Union[str, List[str]],
        media: Union[str, Any, List[Union[str, Any]]],
    ) -> Union[Any, List[Any]]:
        """
        Prepare media as PIL.Image and ensure for every prompt str there is one List[PIL.Image]
        """

        def valid_types(prompts, media):
            from PIL import Image  # type: ignore

            if isinstance(prompts, list):
                if not isinstance(media, list) or len(prompts) != len(media):
                    return False
                for subprompt, submedia in zip(prompts, media):
                    if not isinstance(subprompt, str) or not all(
                        isinstance(m, Image.Image) for m in submedia
                    ):
                        return False
            elif isinstance(prompts, str):
                if not all(isinstance(m, Image.Image) for m in media):
                    return False
            return True

        if not valid_types(prompts, media):
            raise TypeError(
                "Expected (prompts, media) to be of type "
                "(str, List[Image])), or (List[str], List[List[Image]]) "
                f"instead got prompts={prompts}, media={media}"
            )

        return prompts, media
