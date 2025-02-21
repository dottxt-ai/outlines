from functools import singledispatchmethod
from typing import TYPE_CHECKING, List, Tuple, Union

from outlines.models.base import Model, ModelTypeAdapter
from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizer

__all__ = ["Transformers", "Mamba"]


KVCacheType = Tuple[Tuple["torch.DoubleTensor", "torch.DoubleTensor"], ...]


def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:

        class LlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:

        class LlamaTokenizerFast:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:

        class CodeLlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:

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
        from transformers import LogitsProcessorList

        if output_type is not None:
            return LogitsProcessorList([output_type])
        return None


class Transformers(Model):
    """Represents a `transformers` model."""

    def __init__(
        self,
        model_name: str,
        model_class=None,
        model_kwargs: dict = {},
        tokenizer_class=None,
        tokenizer_kwargs: dict = {},
    ):
        """Create a Transformers model instance

        Parameters:
        ----------
        model_name
            The name of the transformers model to use;
        model_class
            The Transformers model class from which to create the model.
            If not provided,`AutoModelForCausalLM` will be used.
            If you gave the name of a non-causal language model,
            you must provide a value for this parameter.
        model_kwargs
            A dictionary of keyword arguments to pass to the `from_pretrained`
            method of the model class.
        tokenizer_class
            The Transformers tokenizer class from which to create the tokenizer.
            If not provided,`AutoTokenizer` will be used.
            If you gave the name of a model that is not compatible with `AutoTokenizer`,
            you must provide a value for this parameter.
        tokenizer_kwargs
            A dictionary of keyword arguments to pass to the `from_pretrained`
            method of the tokenizer class.

        """
        if model_class is None or tokenizer_class is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "The `transformers` library needs to be installed in order to use `transformers` models."
                )
        if model_class is None:
            model_class = AutoModelForCausalLM
        if tokenizer_class is None:
            tokenizer_class = AutoTokenizer
        self.model = model_class.from_pretrained(model_name, **model_kwargs)
        tokenizer_kwargs.setdefault("padding_side", "left")
        self.tokenizer = TransformerTokenizer(
            tokenizer_class.from_pretrained(model_name, **tokenizer_kwargs)
        )
        self.type_adapter = TransformersTypeAdapter()

    def generate(self, model_input, output_type, **inference_kwargs):
        prompts = self.type_adapter.format_input(model_input)
        input_ids, attention_mask = self.tokenizer.encode(prompts)
        inputs = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }

        logits_processor = self.type_adapter.format_output_type(output_type)

        generated_ids = self._generate_output_seq(
            prompts, inputs, logits_processor=logits_processor, **inference_kwargs
        )

        # if single str input, convert to a 1D outputt
        if isinstance(model_input, str):
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def stream(self, model_input, output_type, **inference_kwargs):
        """
        TODO: implement following completion of https://github.com/huggingface/transformers/issues/30810
        """
        raise NotImplementedError(
            "Streaming is not implemented for Transformers models."
        )

    def _generate_output_seq(self, prompts, inputs, **inference_kwargs):
        input_ids = inputs["input_ids"]
        output_ids = self.model.generate(**inputs, **inference_kwargs)

        # encoder-decoder returns output_ids only, decoder-only returns full seq ids
        if self.model.config.is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        # if batch list inputs AND multiple samples per input, convert generated_id to 3D view
        num_samples = inference_kwargs.get("num_return_sequences", 1)
        if num_samples > 1 and isinstance(prompts, list):
            batch_size = input_ids.size(0)
            generated_ids = generated_ids.view(batch_size, num_samples, -1)

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
        else:
            raise TypeError(
                f"Generated outputs aren't 1D, 2D or 3D, but instead are {generated_ids.shape}"
            )


class Mamba(Transformers):
    """Represents a Mamba model."""

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ):
        """
        Create a Mamba model instance

        Parameters:
        ----------
        model_name
            The name of the transformers model to use. It will be passed to
            the `from_pretrained` method of the `MambaForCausalLM` class.
        model_kwargs
            A dictionary of keyword arguments to pass to the `from_pretrained`
            method of the `MambaForCausalLM` class.
        tokenizer_kwargs
            A dictionary of keyword arguments to pass to the `from_pretrained`
            method of the `AutoTokenizer` class.
        """
        try:
            from transformers import MambaForCausalLM

        except ImportError:
            raise ImportError(
                "The `mamba_ssm`, `torch` and `transformer` libraries needs to be installed in order to use Mamba."
            )

        return super().__init__(
            model_name=model_name,
            model_class=MambaForCausalLM,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
