### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings

from outlines.models import VLLMOffline


def vllm_offline_params_adapter(
    sampling_params: dict, generation_params: dict, kwargs: dict
) -> dict:
    """Adapt the parameters of the legacy generator for the `vllm` model
    to the new `Generator` API.
    """
    from vllm.sampling_params import SamplingParams

    # vllm has a `sampling_params` argument that is a `SamplingParams` object
    # this is unrelated to the `sampling_params` argument in the legacy
    # `Generator` API
    sampling_params_kwarg = kwargs.get("sampling_params", SamplingParams())

    # generation_params
    if generation_params.get("stop_at") is not None:
        if isinstance(generation_params["stop_at"], str):
            sampling_params_kwarg.stop_strings = [generation_params["stop_at"]]
        else:
            sampling_params_kwarg.stop_strings = generation_params["stop_at"]

    if generation_params.get("seed") is not None:
        sampling_params_kwarg.seed = generation_params["seed"]

    if generation_params.get("max_tokens") is not None:
        sampling_params_kwarg.max_tokens = generation_params["max_tokens"]

    # sampling_params
    if sampling_params.get("sampler") == "beam_search":
        sampling_params_kwarg.use_beam_search = True

    if (
        sampling_params.get("num_samples") is not None
        and sampling_params_kwarg.n == 1
    ):
        sampling_params_kwarg.n = sampling_params["num_samples"]
        sampling_params_kwarg.best_of = sampling_params["num_samples"]

    if (
        sampling_params.get("top_p") is not None
        and sampling_params_kwarg.top_p == 1.0
    ):
        sampling_params_kwarg.top_p = sampling_params["top_p"]

    if (
        sampling_params.get("top_k") is not None
        and sampling_params_kwarg.top_k == -1
    ):
        sampling_params_kwarg.top_k = sampling_params["top_k"]
        if sampling_params.get("top_k") == 1:
            sampling_params_kwarg.repetition_penalty = 0

    if (
        sampling_params.get("temperature") is not None
        and sampling_params_kwarg.temperature == 1.0
    ):
        sampling_params_kwarg.temperature = sampling_params["temperature"]

    return {"sampling_params": sampling_params_kwarg, **kwargs}


def vllm(model_name: str, **vllm_model_params):
    """Load an offline vLLM model.

    Parameters
    ---------
    model_name
        The name of the model to load from the HuggingFace hub.
    vllm_model_params
        vLLM-specific model parameters. See the vLLM code for the full list:
        https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

    """
    warnings.warn("""
        The `vllm` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Instead, you should instantiate a `VLLMOffline` model with the
        `outlines.from_vllm_offline` function that takes a vLLM model name as
        argument.
        For example:
        ```python
        from vllm import LLM
        from outlines import from_vllm_offline
        model = from_vllm_offline(LLM("<model_name>"))
        ```
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    from vllm import LLM

    model = LLM(model_name, **vllm_model_params)

    return VLLMOffline(model)
