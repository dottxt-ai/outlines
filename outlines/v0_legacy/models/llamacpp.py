### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings
from typing import Optional

from outlines.models import LlamaCpp


def llamacpp_params_adapter(
    sampling_params: dict, generation_params: dict, kwargs: dict
) -> dict:
    """Adapt the parameters of the legacy generator for the `llamacpp` model
    to the new `Generator` API.
    """
    # generation_params
    if (
        generation_params.get("stop_at") is not None
        and kwargs.get("stop") is None
    ):
        kwargs["stop"] = generation_params["stop_at"]

    if (
        generation_params.get("seed") is not None
        and kwargs.get("seed") is None
    ):
        kwargs["seed"] = generation_params["seed"]

    if (
        generation_params.get("max_tokens") is not None
        and kwargs.get("max_tokens") is None
    ):
        kwargs["max_tokens"] = generation_params["max_tokens"]
    else:
        kwargs["max_tokens"] = -1

    # sampling_params
    if sampling_params.get("sampler") == "beam_search":
        raise NotImplementedError(
            "The `llama_cpp_python` library does not support Beam Search."
        )

    if sampling_params.get("num_samples") != 1:
        raise NotImplementedError(
            "The `llama_cpp_python` library does not allow to take several "
            + "samples."
        )

    if (
        sampling_params.get("top_p") is not None
        and kwargs.get("top_p") is None
    ):
        kwargs["top_p"] = sampling_params["top_p"]
    else:
        kwargs["top_p"] = 1.0

    if (
        sampling_params.get("top_k") is not None
        and kwargs.get("top_k") is None
    ):
        kwargs["top_k"] = sampling_params["top_k"]
    else:
        kwargs["top_k"] = -1

    if (
        sampling_params.get("temperature") is not None
        and kwargs.get("temperature") is None
    ):
        kwargs["temperature"] = sampling_params["temperature"]
    else:
        kwargs["temperature"] = 1.0

    # model specific params
    if kwargs.get("min_p") is None:
        kwargs["min_p"] = 0.0

    if kwargs.get("repeat_penalty") is None:
        kwargs["repeat_penalty"] = 1.0

    # The choice to stream or not should happen via the high-level API
    kwargs.pop("stream", None)

    return kwargs


def llamacpp(
    repo_id: str, filename: Optional[str] = None, **llamacpp_model_params
) -> LlamaCpp:
    """Load a model from the `llama-cpp-python` library.

    We use the `Llama.from_pretrained` classmethod that downloads models
    directly from the HuggingFace hub, instead of asking users to specify
    a path to the downloaded model. One can still load a local model
    by initializing `llama_cpp.Llama` directly.

    Parameters
    ----------
    repo_id
        The name of the model repository.
    filename:
        A filename of glob pattern to match the model file in the repo.
    llama_cpp_model_params
        Llama-specific model parameters. See the `llama-cpp-python` documentation
        for the full list: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__

    """
    from llama_cpp import Llama

    warnings.warn("""
        The `llamacpp` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Instead, you should instantiate a `LlamaCpp` model with the
        `outlines.from_llamacpp` function that takes a `llama_cpp.Llama`
        instance as argument. For example:
        ```python
        from llama_cpp import Llama
        from outlines import from_llamacpp
        model = from_llamacpp(Llama.from_pretrained(
            repo_id="<repo_id>",
            filename="<filename>",
        ))
        ```
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    # Default to using the model's full context length
    if "n_ctx" not in llamacpp_model_params:
        llamacpp_model_params["n_ctx"] = 0

    if "verbose" not in llamacpp_model_params:
        llamacpp_model_params["verbose"] = False

    # TODO: Remove when https://github.com/ggerganov/llama.cpp/pull/5613 is resolved
    if "tokenizer" not in llamacpp_model_params:
        warnings.warn(
            "The pre-tokenizer in `llama.cpp` handles unicode improperly "
            + "(https://github.com/ggerganov/llama.cpp/pull/5613)\n"
            + "Outlines may raise a `RuntimeError` when building the regex index.\n"
            + "To circumvent this error when using `models.llamacpp()` you may pass the argument"
            + "`tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(<hf_repo_id>)`\n"
        )

    model = Llama.from_pretrained(repo_id, filename, **llamacpp_model_params)

    return LlamaCpp(model)
