"""Integration with the `transformers` library. """

import warnings

from collections import defaultdict
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from outlines.inputs import Audio, Chat, Image, Video
from outlines.models.base import Model, ModelTypeAdapter
from outlines.models.tokenizer import Tokenizer
from outlines.processors import OutlinesLogitsProcessor

if TYPE_CHECKING:
    import torch
    from transformers import (
        PreTrainedTokenizer,
        PreTrainedModel,
        ProcessorMixin,
        LogitsProcessorList,
    )

__all__ = ["Transformers", "TransformersMultiModal", "from_transformers"]


def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:  # pragma: no cover

        class LlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:  # pragma: no cover

        class LlamaTokenizerFast:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:  # pragma: no cover

        class CodeLlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:  # pragma: no cover

        class CodeLlamaTokenizerFast:  # type: ignore
            pass

    return (
        LlamaTokenizer,
        LlamaTokenizerFast,
        CodeLlamaTokenizer,
        CodeLlamaTokenizerFast,
    )


class TransformerTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: "PreTrainedTokenizer", **kwargs):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple["torch.LongTensor", "torch.LongTensor"]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: "torch.LongTensor") -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = self.tokenizer.convert_tokens_to_string([token])

        if self.is_llama:
            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, "model_name") and hasattr(self, "kwargs"):
                return (
                    other.model_name == self.model_name and other.kwargs == self.kwargs
                )
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __hash__(self):
        from datasets.fingerprint import Hasher

        return hash(Hasher.hash(self.tokenizer))

    def __getstate__(self):
        state = {"tokenizer": self.tokenizer}
        return state

    def __setstate__(self, state):
        self.__init__(state["tokenizer"])


class TransformersTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `Transformers` model."""

    def __init__(self, **kwargs):
        self.tokenizer = kwargs.get("tokenizer")

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the prompt argument to pass to the model.

        Parameters
        ----------
        model_input
            The input passed by the user.

        Returns
        -------
        str
            The formatted input to be passed to the model.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available."
            "The only available types are `str` and `Chat`."
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str) -> str:
        return model_input

    @format_input.register(Chat)
    def format_chat_input(self, model_input: Chat) -> str:
        return self.tokenizer.apply_chat_template(
            model_input.messages,
            tokenize=False
        )

    def format_output_type(
        self,
        output_type: Optional[OutlinesLogitsProcessor] = None,
    ) -> Optional["LogitsProcessorList"]:
        """Generate the logits processor argument to pass to the model.

        Parameters
        ----------
        output_type
            The logits processor provided.

        Returns
        -------
        Optional[LogitsProcessorList]
            The logits processor to pass to the model.

        """
        from transformers import LogitsProcessorList

        if output_type is not None:
            return LogitsProcessorList([output_type])
        return None


class Transformers(Model):
    """Thin wrapper around a `transformers` model and a `transformers`
    tokenizer.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `transformers` model and
    tokenizer.

    """

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        """
        Parameters:
        ----------
        model
            A `PreTrainedModel`, or any model that is compatible with the
            `transformers` API for models.
        tokenizer
            A `PreTrainedTokenizer`, or any tokenizer that is compatible with
            the `transformers` API for tokenizers.

        """
        # We need to handle the cases in which jax/flax or tensorflow
        # is not available in the environment.
        try:
            from transformers import FlaxPreTrainedModel
        except ImportError:  # pragma: no cover
            FlaxPreTrainedModel = None

        try:
            from transformers import TFPreTrainedModel
        except ImportError:  # pragma: no cover
            TFPreTrainedModel = None

        tokenizer.padding_side = "left"
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)
        self.type_adapter = TransformersTypeAdapter(tokenizer=tokenizer)

        if (
            FlaxPreTrainedModel is not None
            and isinstance(model, FlaxPreTrainedModel)
        ):
            self.tensor_library_name = "jax"
        elif (
            TFPreTrainedModel is not None
            and isinstance(model, TFPreTrainedModel)
        ):
            self.tensor_library_name = "tensorflow"
        else:
            self.tensor_library_name = "torch"

    def _prepare_model_inputs(
        self,
        model_input,
        is_batch: bool = False,
    ) -> Tuple[Union[str, List[str]], dict]:
        """Turn the user input into arguments to pass to the model"""
        # Format validation
        if is_batch:
            prompts = [
                self.type_adapter.format_input(item)
                for item in model_input
            ]
        else:
            prompts = self.type_adapter.format_input(model_input)
        input_ids, attention_mask = self.tokenizer.encode(prompts)
        inputs = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }

        return prompts, inputs

    def generate(
        self,
        model_input: Union[str, dict, Chat],
        output_type: Optional[OutlinesLogitsProcessor] = None,
        **inference_kwargs: Any,
    ) -> Union[str, List[str]]:
        """Generate text using `transformers`.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response. For
            multi-modal models, the input should be a dictionary containing the
            `text` key with a value of type `Union[str, List[str]]` and the
            other keys required by the model.
        output_type
            The logits processor the model will use to constrain the format of
            the generated text.
        inference_kwargs
            Additional keyword arguments to pass to the `generate` method
            of the `transformers` model.

        Returns
        -------
        Union[str, List[str]]
            The text generated by the model.

        """
        prompts, inputs = self._prepare_model_inputs(model_input, False)
        logits_processor = self.type_adapter.format_output_type(output_type)

        generated_ids = self._generate_output_seq(
            prompts,
            inputs,
            logits_processor=logits_processor,
            **inference_kwargs,
        )

        # required for multi-modal models that return a 2D tensor even when
        # num_return_sequences is 1
        num_samples = inference_kwargs.get("num_return_sequences", 1)
        if num_samples == 1 and len(generated_ids.shape) == 2:
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def generate_batch(
        self,
        model_input: List[Union[str, dict, Chat]],
        output_type: Optional[OutlinesLogitsProcessor] = None,
        **inference_kwargs: Any,
    ) -> List[Union[str, List[str]]]:
        """"""
        prompts, inputs = self._prepare_model_inputs(model_input, True) # type: ignore
        logits_processor = self.type_adapter.format_output_type(output_type)

        generated_ids = self._generate_output_seq(
            prompts, inputs, logits_processor=logits_processor, **inference_kwargs
        )

        # if there are multiple samples per input, convert generated_id to 3D
        num_samples = inference_kwargs.get("num_return_sequences", 1)
        if num_samples > 1:
            generated_ids = generated_ids.view(len(model_input), num_samples, -1)

        return self._decode_generation(generated_ids)

    def generate_stream(self, model_input, output_type, **inference_kwargs):
        """Not available for `transformers` models.

        TODO: implement following completion of https://github.com/huggingface/transformers/issues/30810

        """
        raise NotImplementedError(
            "Streaming is not implemented for Transformers models."
        )

    def _generate_output_seq(self, prompts, inputs, **inference_kwargs):
        input_ids = inputs["input_ids"]

        output_ids = self.model.generate(
            **inputs,
            **inference_kwargs,
        )

        # encoder-decoder returns output_ids only, decoder-only returns full seq ids
        if self.model.config.is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        return generated_ids

    def _decode_generation(self, generated_ids: "torch.Tensor"):
        if len(generated_ids.shape) == 1:
            return self.tokenizer.decode([generated_ids])[0]
        elif len(generated_ids.shape) == 2:
            return self.tokenizer.decode(generated_ids)
        elif len(generated_ids.shape) == 3:
            return [
                self.tokenizer.decode(generated_ids[i])
                for i in range(len(generated_ids))
            ]
        else:  # pragma: no cover
            raise TypeError(
                "Generated outputs aren't 1D, 2D or 3D, but instead are "
                f"{generated_ids.shape}"
            )


class TransformersMultiModalTypeAdapter(ModelTypeAdapter):
    """Type adapter for `TransformersMultiModal` model."""

    def __init__(self, **kwargs):
        self.tokenizer = kwargs.get("tokenizer")

    @singledispatchmethod
    def format_input(self, model_input):
        """Fomat the prompt arguments to pass to the model.

        Argument
        --------
        model_input
            The input passed by the user.

        Returns
        -------
        dict
            The formatted input.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available. Please "
            + "provide a list containing a text prompt and assets "
            + "(`Image`, `Audio` or `Video` instances) supported by your "
            + "model or a `Chat` instance."
        )

    @format_input.register(dict)
    def format_dict_input(self, model_input: dict) -> dict:
        warnings.warn("""
            Providing the input as a dict is deprecated. Support for this will
            be removed in the v1.2.0 release of Outlines. Use a list containing
            a text prompt and assets (`Image`, `Audio` or `Video` instances)
            instead.
            For instance:
            ```python
            from outlines import Image
            model = from_transformers(mymodel, myprocessor)
            response = model([
                "A beautiful image of a cat",
                Image(my_image),
            ])
            ```
            """,
            DeprecationWarning,
            stacklevel=2,
        )
        if "text" not in model_input:
            raise ValueError(
                "The input must contain the 'text' key along with the other "
                + "keys required by your processor."
            )
        return model_input

    @format_input.register(Chat)
    def format_chat_input(self, model_input: Chat) -> dict:
        # we need to separate the images from the messages
        # to apply the chat template to the messages without images
        messages = model_input.messages
        images = []
        messages_without_images = []
        for message in messages:
            if isinstance(message["content"], list):
                images.extend(message["content"][1:])
                messages_without_images.append({
                    "role": message["role"],
                    "content": message["content"][0],
                })
            else:
                messages_without_images.append(message)
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages_without_images,
            tokenize=False
        )
        # use the formatted prompt and the images to format the input
        return self.format_list_input([formatted_prompt, *images])

    @format_input.register(list)
    def format_list_input(self, model_input: list) -> dict:
        prompt = model_input[0]
        assets = model_input[1:]

        asset_types = set(type(asset) for asset in assets)
        if len(asset_types) > 1:
            raise ValueError(
                "All assets must be of the same type. "
                + f"Found types: {asset_types}"
            )
        asset_type = asset_types.pop()

        if asset_type == Image:
            return {
                "text": prompt,
                "images": [asset.image for asset in assets]
            }
        elif asset_type == Audio: # pragma: no cover
            return {
                "text": prompt,
                "audio": [asset.audio for asset in assets]
            }
        elif asset_type == Video: # pragma: no cover
            return {
                "text": prompt,
                "videos": [asset.video for asset in assets]
            }
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")

    def format_output_type(
        self,
        output_type: Optional[OutlinesLogitsProcessor] = None,
    ) -> Optional["LogitsProcessorList"]:
        """Generate the logits processor argument to pass to the model.

        Argument
        --------
        output_type
            The logits processor provided.

        Returns
        -------
        Optional[LogitsProcessorList]
            The logits processor to pass to the model.

        """
        from transformers import LogitsProcessorList

        if output_type is not None:
            return LogitsProcessorList([output_type])
        return None


class TransformersMultiModal(Transformers):
    """Thin wrapper around a `transformers` model and a `transformers`
    processor.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `transformers` model and
    processor.

    """

    def __init__(self, model: "PreTrainedModel", processor):
        """Create a TransformersMultiModal model instance

        We rely on the `__init__` method of the `Transformers` class to handle
        most of the initialization and then add elements specific to multimodal
        models.

        Parameters
        ----------
        model
            A `PreTrainedModel`, or any model that is compatible with the
            `transformers` API for models.
        processor
            A `ProcessorMixin` instance.

        """
        self.processor = processor
        self.processor.padding_side = "left"
        self.processor.pad_token = "[PAD]"

        tokenizer: "PreTrainedTokenizer" = self.processor.tokenizer

        super().__init__(model, tokenizer)

        self.type_adapter = TransformersMultiModalTypeAdapter(
            tokenizer=tokenizer
        )

    def _prepare_model_inputs(
        self,
        model_input,
        is_batch: bool = False,
    ) -> Tuple[Union[str, List[str]], dict]:
        """Turn the user input into arguments to pass to the model"""
        if is_batch:
            prompts = [
                self.type_adapter.format_input(item) for item in model_input
            ]
        else:
            prompts = self.type_adapter.format_input(model_input)

        # The expected format is a single dict
        if is_batch:
            merged_prompts = defaultdict(list)
            for d in prompts:
                for key, value in d.items():
                    if key == "text":
                        merged_prompts[key].append(value)
                    else:
                        merged_prompts[key].extend(value)
        else:
            merged_prompts = prompts # type: ignore

        inputs = self.processor(
            **merged_prompts, padding=True, return_tensors="pt"
        ).to(self.model.device)

        return merged_prompts["text"], inputs


def from_transformers(
    model: "PreTrainedModel",
    tokenizer_or_processor: Union["PreTrainedTokenizer", "ProcessorMixin"],
) -> Union[Transformers, TransformersMultiModal]:
    """Create an Outlines `Transformers` or `TransformersMultiModal` model
    instance from a `PreTrainedModel` instance and a `PreTrainedTokenizer` or
    `ProcessorMixin` instance.

    `outlines` supports `PreTrainedModelForCausalLM`,
    `PreTrainedMambaForCausalLM`, `PreTrainedModelForSeq2Seq` and any model
    that implements the `transformers` model API.

    Parameters
    ----------
    model
        A `transformers.PreTrainedModel` instance.
    tokenizer_or_processor
        A `transformers.PreTrainedTokenizer` or
        `transformers.ProcessorMixin` instance.

    Returns
    -------
    Union[Transformers, TransformersMultiModal]
        An Outlines `Transformers` or `TransformersMultiModal` model instance.

    """
    from transformers import (
        PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin)

    if isinstance(
        tokenizer_or_processor, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    ):
        tokenizer = tokenizer_or_processor
        return Transformers(model, tokenizer)
    elif isinstance(tokenizer_or_processor, ProcessorMixin):
        processor = tokenizer_or_processor
        return TransformersMultiModal(model, processor)
    else:
        raise ValueError(
            "We could determine whether the model passed to `from_transformers`"
            + " is a text-2-text or a multi-modal model. Please provide a "
            + "a transformers tokenizer or processor."
        )
