import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


__all__ = ["transformers"]


class Transformers:
    """Represents a `transformers` model."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        device: Optional[str] = None,
    ):
        self.device = device if device is not None else "cpu"
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: NDArray[np.int64], attention_mask: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        import torch

        # `transformers` model accept `input_ids` of size at most equal to 2. We
        # thus reshape the input array, call the model and reshape the output
        # logits.
        batch_shape = input_ids.shape[:-1]
        num_tokens = input_ids.shape[-1]
        input_ids = input_ids.reshape(math.prod(batch_shape), num_tokens)

        with torch.no_grad():
            input_ids = torch.from_numpy(input_ids).to(self.device)
            attention_mask = torch.from_numpy(attention_mask).to(self.device)

            output = self.model(input_ids, attention_mask=attention_mask)

            next_token_logits = output.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1).squeeze()
            probs = torch.atleast_2d(probs)
            numpy_probs = probs.cpu().detach().numpy()

        return numpy_probs.reshape(batch_shape + (-1,))


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

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "np"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: NDArray[np.int64]) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids)
        return text


def transformers(model_name: str, device: Optional[str] = None, **model_kwargs):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = TransformersTokenizer(model_name)

    return Transformers(model, tokenizer, device)
