import functools
import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from outlines.models.transformers import TransformersTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


__all__ = ["random"]


class RandomModel:
    """Represents a `random` model, that samples from a given distribution."""

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        seed: Optional[int] = None,
        dist: str = "uniform",
        device: Optional[str] = None,
    ):
        self.device = device if device is not None else "cpu"
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
        self.tokenizer = tokenizer
        if dist == "uniform":
            self.model = functools.partial(
                torch.rand, generator=self.rng, device=self.device
            )
        else:
            raise NotImplementedError("Only Uniform distribution supported as of now.")

    def __call__(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        batch_shape = input_ids.shape[:-1]
        num_tokens = input_ids.shape[-1]
        input_ids = input_ids.reshape(math.prod(batch_shape), num_tokens)

        # Sample from the distribution to get the logits
        next_token_logits = self.model(
            size=(input_ids.shape[0], self.tokenizer.tokenizer.vocab_size)
        )

        # Reshape the logits to match the input shape
        next_token_logits = next_token_logits.reshape(batch_shape + (-1,))

        return next_token_logits


def random(
    tokenizer: str,
    dist: str = "uniform",
    seed: Optional[int] = None,
    device: Optional[str] = None,
):
    tokenizer = TransformersTokenizer(tokenizer)

    return RandomModel(tokenizer, seed, dist, device)
