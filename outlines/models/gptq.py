from typing import Optional

from .transformers import Transformer, TransformerTokenizer


def gptq(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except ImportError:
        raise ImportError(
            "The `auto-gptq` library needs to be installed in order to use `AutoGPTQ` models."
        )

    if device is not None:
        model_kwargs["device_map"] = device

    model = AutoGPTQForCausalLM.from_quantized(model_name, **model_kwargs)
    tokenizer = TransformerTokenizer(model_name, **tokenizer_kwargs)

    return Transformer(model, tokenizer)
