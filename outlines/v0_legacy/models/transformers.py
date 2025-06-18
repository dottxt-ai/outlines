### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings
from typing import Optional

from outlines.models import Transformers


def transformers_params_adapter(
    sampling_params: dict, generation_params: dict, kwargs: dict
) -> dict:
    """Adapt the parameters of the legacy generator for the `transformers` model
    to the new `Generator` API.
    """
    from transformers import set_seed

    # generation_params
    if (
        generation_params.get("stop_at") is not None
        and kwargs.get("stop_strings") is None
    ):
        kwargs["stop_strings"] = generation_params["stop_at"]

    if (
        generation_params.get("max_tokens") is not None
        and kwargs.get("max_new_tokens") is None
    ):
        kwargs["max_new_tokens"] = generation_params["max_tokens"]
    else:
        kwargs["max_new_tokens"] = int(2**30)

    if (
        generation_params.get("seed") is not None
        or kwargs.get("seed") is not None
    ):
        seed = (
            generation_params.get("seed")
            if generation_params.get("seed") is not None
            else kwargs.get("seed")
        )
        set_seed(seed) # global seed

    # sampling_params
    if sampling_params.get("sampler") == "multinomial":
        kwargs["do_sample"] = True
    elif sampling_params.get("sampler") == "beam_search":
        kwargs["num_beams"] = sampling_params.get("num_samples", 1)

    if (
        sampling_params.get("num_samples") is not None
        and kwargs.get("num_return_sequences") is None
    ):
        kwargs["num_return_sequences"] = sampling_params["num_samples"]
    else:
        kwargs["num_return_sequences"] = 1

    if (
        sampling_params.get("top_p") is not None
        and kwargs.get("top_p") is None
    ):
        kwargs["top_p"] = sampling_params["top_p"]

    if (
        sampling_params.get("top_k") is not None
        and kwargs.get("top_k") is None
    ):
        kwargs["top_k"] = sampling_params["top_k"]

    if (
        sampling_params.get("temperature") is not None
        and kwargs.get("temperature") is None
    ):
        kwargs["temperature"] = sampling_params["temperature"]

    return kwargs


def transformers(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
    model_class=None,
    tokenizer_class=None,
):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    device
        The device(s) on which the model should be loaded. This overrides
        the `device_map` entry in `model_kwargs` when provided.
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
    warnings.warn("""
        The `transformers` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Instead, you should instantiate a `Transformers` model with the
        `outlines.from_transformers` function that takes a transformers model
        and a tokenizer as arguments. For example:
        ```python
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from outlines import from_transformers
        model = from_transformers(
            AutoModelForCausalLM.from_pretrained("<model_name>"),
            AutoTokenizer.from_pretrained("<model_name>")
        )
        ```
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if model_class is None or tokenizer_class is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:  # pragma: no cover
            raise ImportError(
                "The `transformers` library needs to be installed in order to use `transformers` models."
            )
    if model_class is None:
        model_class = AutoModelForCausalLM
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    if device is not None:
        model_kwargs["device_map"] = device

    model = model_class.from_pretrained(model_name, **model_kwargs)

    tokenizer_kwargs.setdefault("padding_side", "left")
    tokenizer = tokenizer_class.from_pretrained(model_name, **tokenizer_kwargs)

    return Transformers(model, tokenizer)


def mamba(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    warnings.warn("""
        The `mamba` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Instead, you should instantiate a `Transformers` model with the
        `outlines.from_transformers` function that takes a transformers model
        and a tokenizer as arguments. For Mamba, you would use a mamba model
        and a mamba tokenizer. For example:
        ```python
        from transformers import MambaForCausalLM, MambaTokenizer
        from outlines import from_transformers
        model = from_transformers(
            MambaForCausalLM.from_pretrained("<repo_id>"),
            MambaTokenizer.from_pretrained("<repo_id>")
        )
        ```
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        from transformers import MambaForCausalLM

    except ImportError:  # pragma: no cover
        raise ImportError(
            "The `mamba_ssm`, `torch` and `transformer` libraries needs to be installed in order to use Mamba."
        )

    # We do not want to raise the deprecation warning through the transformers function
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
        )
        return transformers(
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            model_class=MambaForCausalLM,
    )
