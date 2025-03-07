import dataclasses
from functools import singledispatchmethod
from typing import TYPE_CHECKING, List, Optional, Union

from outlines.models.base import Model, ModelTypeAdapter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM
    from vllm.sampling_params import SamplingParams


__all__ = ["VLLM", "from_vllm"]


class VLLMTypeAdapter(ModelTypeAdapter):
    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the prompt argument to pass to the model.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        raise NotImplementedError(
            f"The input type {input} is not available."
            "Please use a string or a list of strings."
        )

    @format_input.register(str)
    def format_str_input(self, model_input):
        return model_input

    @format_input.register(list)
    def format_list_input(self, model_input):
        return model_input

    def format_output_type(self, output_type):
        """Generate the logits processor argument to pass to the model.

        Argument
        --------
        output_type
            The logits processor provided.

        """
        if output_type is not None:
            return [output_type]
        return []


class VLLM(Model):
    """Represents a `vllm` model."""

    def __init__(self, model: "LLM"):
        """Create a VLLM model instance.

        Parameters
        ----------
        model
            A `vllm.LLM` model instance.

        """
        self.model = model
        self.tokenizer = self._get_tokenizer()
        self.type_adapter = VLLMTypeAdapter()

    def _get_tokenizer(self):
        if hasattr(self.model, "get_tokenizer"):
            tokenizer = self.model.get_tokenizer()
        elif hasattr(self.model, "tokenizer"):
            if hasattr(self.model.tokenizer, "tokenizer"):
                tokenizer = self.model.tokenizer.tokenizer
            else:
                tokenizer = self.model.tokenizer
        else:
            raise ValueError(
                "The provided LLM instance neither has a "
                "`tokenizer` attribute or a `get_tokenizer` method."
            )
        return adapt_tokenizer(tokenizer=tokenizer)

    def generate(self, model_input, output_type, **inference_kwargs):
        """Generate text using `vllm`.

        Arguments
        ---------
        prompt
            The text prompt provided by the user.
        logits_processor
            The logits processor to use when generating text.
        inference_kwargs
            The inference kwargs that can be passed to the `LLM.generate` method
            in the `vllm` library.

        Returns
        -------
        The generated text.

        """
        from vllm.sampling_params import SamplingParams

        sampling_params = inference_kwargs.pop("sampling_params", None)
        if sampling_params is None:
            sampling_params = SamplingParams()
        sampling_params.logits_processors = self.type_adapter.format_output_type(output_type)

        results = self.model.generate(
            self.type_adapter.format_input(model_input),
            sampling_params=sampling_params,
            **inference_kwargs,
        )
        results = [[sample.text for sample in batch.outputs] for batch in results]

        batch_size = len(results)
        sample_size = len(results[0])

        if batch_size == 1 and sample_size == 1:
            return results[0][0]
        elif batch_size == 1:
            return results[0]
        elif sample_size == 1:
            return [batch[0] for batch in results]

        return results

    def generate_stream(self, model_input, output_type, **inference_kwargs):
        """Return a text generator.

        Streaming is not yet available for `vllm.LLM`.

        TODO: Implement the streaming functionality ourselves.

        """
        raise NotImplementedError(
            "Streaming is not available for the vLLM integration."
        )


def from_vllm(model: "LLM") -> VLLM:
    return VLLM(model)


def adapt_tokenizer(tokenizer: "PreTrainedTokenizerBase") -> "PreTrainedTokenizerBase":
    """Adapt a tokenizer to use to compile the FSM.

    The API of Outlines tokenizers is slightly different to that of `transformers`. In
    addition we need to handle the missing spaces to Llama's tokenizer to be able to
    compile FSMs for this model.

    Parameters
    ----------
    tokenizer
        The tokenizer of the model.

    Returns
    -------
    PreTrainedTokenizerBase
        The adapted tokenizer.
    """
    from transformers import SPIECE_UNDERLINE

    tokenizer.vocabulary = tokenizer.get_vocab()
    tokenizer.special_tokens = set(tokenizer.all_special_tokens)

    def convert_token_to_string(token: Union[str, bytes]) -> str:
        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (
            type(token) is str
            and token.startswith(SPIECE_UNDERLINE)
            or token == "<0x20>"
        ):
            return " " + string

        return string

    tokenizer.convert_token_to_string = convert_token_to_string

    return tokenizer
