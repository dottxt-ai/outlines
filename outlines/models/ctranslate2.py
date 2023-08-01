import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ctranslate2 import Generator


__all__ = ["ctranslate2"]


class CTranslate2_Model:
    """Represents a `ctranslate2` model."""

    def __init__(
        self,
        model: "Generator",
        tokenizer: "PreTrainedTokenizer",
        device: Optional[str] = None,
    ):
        self.device = device if device is not None else "cpu"
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        # `forward_batch` method of `Generator` accepts `tokens` in a list of list of str
        tokens = [self.tokenizer.tokenizer.convert_ids_to_tokens(iids) for iids in input_ids]
        logits = self.model.forward_batch([tokens], return_log_probs=True)
        logits = torch.as_tensor(logits)
        next_token_logits = logits[:, -1, :]

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
        return string


def ctranslate2(
    ctr2_model: str, tokenizer_name: str, device: Optional[str] = None, **model_kwargs
):
    import ctranslate2

    model = ctranslate2.Generator(ctr2_model, device=device)
    tokenizer = TransformersTokenizer(tokenizer_name, **model_kwargs)

    return CTranslate2_Model(model, tokenizer)
