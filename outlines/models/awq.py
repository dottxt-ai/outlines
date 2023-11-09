from typing import TYPE_CHECKING, Optional

from .transformers import Transformer, TransformerTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class AWQModel(Transformer):
    """Represents a `transformers` model."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.device = model.model.device
        self.model = model
        self.tokenizer = tokenizer


def awq(
    model_name: str,
    fuse_layers: bool = True,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError(
            "The `autoawq` and `transformers` library needs to be installed in order to use `AutoAWQ` models."
        )

    model_kwargs["fuse_layers"] = fuse_layers
    model_kwargs["safetensors"] = True

    if device is not None:
        model_kwargs["device_map"] = device

    model = AutoAWQForCausalLM.from_quantized(model_name, **model_kwargs)
    tokenizer = TransformerTokenizer(model_name, trust_remote_code=True)

    return AWQModel(model, tokenizer)
