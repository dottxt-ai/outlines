import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from transformers.file_utils import SPIECE_UNDERLINE

from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = ["transformers"]


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


class Transformers:
    """Represents a `transformers` model."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        # `transformers` model accept `input_ids` of size at most equal to 2. We
        # thus reshape the input array, call the model and reshape the output
        # logits.
        batch_shape = input_ids.shape[:-1]
        num_tokens = input_ids.shape[-1]
        input_ids = input_ids.reshape(math.prod(batch_shape), num_tokens)

        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        next_token_logits = output.logits[:, -1, :]

        next_token_logits = next_token_logits.reshape(batch_shape + (-1,))

        return next_token_logits


class TransformersTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, model_name: str, **kwargs):
        from transformers import AutoTokenizer

        kwargs.setdefault("padding_side", "left")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: torch.LongTensor) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids)
        return text

    def convert_token_to_string(self, token: str) -> str:
        string = self.tokenizer.convert_tokens_to_string([token])

        if self.is_llama:
            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string


def transformers(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    device_map
        The device(s) on which the model should be loaded. This overrides
        the value passed for `device_map` in `model_kwargs`.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the model.
    tokenizer_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the tokenizer.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "The `transformers` library needs to be installed in order to use `transformers` models."
        )

    model_kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = TransformersTokenizer(model_name, **tokenizer_kwargs)

    return Transformers(model, tokenizer)
