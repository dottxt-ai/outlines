### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

from typing import Optional
import warnings

from outlines.models import MLXLM


def mlxlm_params_adapter(
    sampling_params: dict, generation_params: dict, kwargs: dict
) -> dict:
    """Adapt the parameters of the legacy generator for the `mlxlm` model
    to the new `Generator` API.
    """
    # generation_params
    if generation_params.get("stop_at") is not None:
        raise NotImplementedError(
            "The `mlx-lm` library does not support `stop_at`."
        )

    if generation_params.get("seed") is not None:
        raise NotImplementedError(
            "The `mlx-lm` library does not support `seed`."
        )

    if (
        generation_params.get("max_tokens") is not None
        and kwargs.get("max_tokens") is None
    ):
        kwargs["max_tokens"] = generation_params["max_tokens"]
    else:
        kwargs["max_tokens"] = int(1e9)

    # sampling_params
    if sampling_params.get("sampler") == "beam_search":
        raise NotImplementedError(
            "The `mlx-lm` library does not support Beam Search."
        )

    if sampling_params.get("num_samples") != 1:
        raise NotImplementedError(
            "The `mlx-lm` library does not allow to take several samples."
        )

    if (
        sampling_params.get("top_k") is not None
        or kwargs.get("top_k") is not None
    ):
        raise NotImplementedError(
            "The `mlx-lm` library does not support top_k."
        )

    if (
        sampling_params.get("top_p") is not None
        or kwargs.get("top_p") is not None
    ):
        kwargs.pop("top_p", None)
        warnings.warn(
            """
            The `top_p` parameter is not available anymore.
            The argument will be ignored. Sorry for the inconvenience.
            Please migrate to the v1 of Outlines.
            """,
            DeprecationWarning,
            stacklevel=2,
        )

    if (
        sampling_params.get("temperature") is not None
        or kwargs.get("temperature") is not None
    ):
        kwargs.pop("temperature", None)
        warnings.warn(
            """
            The `temperature` parameter is not available anymore.
            The argument will be ignored. Sorry for the inconvenience.
            Please migrate to the v1 of Outlines.
            """,
            DeprecationWarning,
            stacklevel=2,
        )

    return kwargs


def mlxlm(
    model_name: str,
    tokenizer_config: dict = {},
    model_config: dict = {},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
):
    """Instantiate a model from the `mlx_lm` library and its tokenizer.

    Signature adapted from
    https://github.com/ml-explore/mlx-examples/blob/4872727/llms/mlx_lm/utils.py#L422

    Parameters
    ----------
    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``

    Returns
    -------
    A `MLXLM` model instance.

    """
    warnings.warn("""
        The `mlxlm` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Instead, you should instantiate a `MLXLM` model with the
        `outlines.from_mlxlm` function that takes a mlxlm model
        and a tokenizer as arguments. For example:
        ```python
        from mlx_lm import load
        from outlines import from_mlxlm
        model = from_mlxlm(
            load("<model_name>")
        )
        ```
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        raise ImportError(
            "The `mlx_lm` library needs to be installed in order to use `mlx_lm` models."
        )
    if not mx.metal.is_available():
        raise RuntimeError("You cannot use `mlx_lm` without Apple Silicon (Metal)")

    model, tokenizer = mlx_lm.load(
        model_name,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        adapter_path=adapter_path,
        lazy=lazy,
    )
    return MLXLM(model, tokenizer)
