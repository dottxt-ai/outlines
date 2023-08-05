import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

import functools
from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


__all__ = ["random"]


class RandomModel:
    """Represents a `random` model, that samples from a given distribution everytime"""

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        seed: Optional[int] = None,
        dist: str = 'uniform',
        device: Optional[str] = None,
    ):
        self.device = device if device is not None else "cpu"
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
        self.tokenizer = tokenizer
        if dist == 'uniform':
            self.model = functools.partial(torch.rand, generator=self.rng, device=self.device)
        else:
            raise NotImplementedError("Only Uniform distribution supported as of now.")

    def __call__(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        batch_shape = input_ids.shape[:-1]
        num_tokens = input_ids.shape[-1]
        input_ids = input_ids.reshape(math.prod(batch_shape), num_tokens)

        # Sample from the distribution to get the logits
        next_token_logits = self.model(size=(input_ids.shape[0], self.tokenizer.tokenizer.vocab_size))

        # Reshape the logits to match the input shape
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


def random(tokenizer: str, dist: str = 'uniform', seed: Optional[int] = None, device: Optional[str] = None):
    tokenizer = TransformersTokenizer(tokenizer)

    return RandomModel(tokenizer, seed, dist, device)
