### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from outlines.generator import Generator
from outlines.v0_legacy.models.exllamav2 import ExLlamaV2Model, exllamav2_params_adapter
from outlines.v0_legacy.models.transformers import transformers_params_adapter
from outlines.v0_legacy.models.transformers_vision import TransformersVision
from outlines.v0_legacy.models.llamacpp import llamacpp_params_adapter
from outlines.v0_legacy.models.vllm_offline import vllm_offline_params_adapter
from outlines.v0_legacy.models.mlxlm import mlxlm_params_adapter
from outlines.models import (
    LlamaCpp,
    MLXLM,
    Transformers,
    VLLMOffline,
)

V0GeneratorModel = Union[
    ExLlamaV2Model,
    LlamaCpp,
    MLXLM,
    Transformers,
    TransformersVision,
    VLLMOffline,
]


class GeneratorV0Adapter:
    """Object used to bridge the gap between the legacy `generate` functions
    and the current `Generator` object.

    The legacy `generate` functions return an instance of this class. When
    initialized, it stores an instance of the current `Generator` object
    and calls it when it's called.
    """

    def __init__(self, model, output_type, sampler):
        if not isinstance(model, V0GeneratorModel):  # type: ignore
            raise ValueError(
                "You can only use the v0 API with models that were already"
                + f"available in v0. Got {type(model)}."
            )
        self.model = model
        self.generator = Generator(model, output_type)
        self.sampling_params = asdict(sampler.sampling_params)

    def format_sequence(self, sequence: str):
        return sequence

    def create_inference_params(
        self, max_tokens, stop_at, seed, **model_specific_params
    ):
        """Turn the arguments received by the generator into arguments
        for the model associated to it.
        """
        generation_params = {
            "max_tokens": max_tokens,
            "stop_at": stop_at,
            "seed": seed,
        }

        if isinstance(self.model, LlamaCpp):
            adapter_function = llamacpp_params_adapter
        elif isinstance(self.model, VLLMOffline): # pragma: no cover
            adapter_function = vllm_offline_params_adapter
        elif isinstance(self.model, Transformers):
            adapter_function = transformers_params_adapter
        elif isinstance(self.model, ExLlamaV2Model): # pragma: no cover
            adapter_function = exllamav2_params_adapter
        elif isinstance(self.model, MLXLM): # pragma: no cover
            adapter_function = mlxlm_params_adapter

        inference_params = adapter_function(
            self.sampling_params,
            generation_params,
            model_specific_params
        )
        return inference_params

    def __call__(
        self,
        prompts: Union[str, List[str], Dict[str, Any]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        result = self.generator(
            prompts,
            **self.create_inference_params(
                max_tokens, stop_at, seed, **model_specific_params
            )
        )
        if isinstance(result, list):
            return [self.format_sequence(r) for r in result]
        else:
            return self.format_sequence(result)

    def stream(
        self,
        prompts: Union[str, List[str], Dict[str, Any]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        return self.generator.stream(
            prompts,
            **self.create_inference_params(
                max_tokens, stop_at, seed, **model_specific_params
            )
        )


class GeneratorVisionV0Adapter(GeneratorV0Adapter):
    """This object does the same as `GeneratorV0Adapter` but for the
    `TransformersVision` model. The only difference between the two is that
    it receives a media argument in addition to the prompts.
    """

    @staticmethod
    def merge_prompts_and_media(
        prompts: Union[str, List[str]],
        media: Union[Any, List[Any]],
    ) -> Dict[str, Any]:
        """Merge prompts and media into a single dictionary

        This function is needed as the `TransformersMultiModal` model requires
        the prompt and assets to be contained in a single argument while the
        legacy `TransformersVision` model requires separate arguments for the
        prompt and assets.
        """
        return {
            "text": prompts,
            "images": media,
        }

    def __call__(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        merged_input = self.merge_prompts_and_media(prompts, media)
        return super().__call__(
            merged_input,
            max_tokens,
            stop_at,
            seed,
            **model_specific_params
        )

    def stream(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        merged_input = self.merge_prompts_and_media(prompts, media)
        return super().stream(
            merged_input,
            max_tokens,
            stop_at,
            seed,
            **model_specific_params
        )
