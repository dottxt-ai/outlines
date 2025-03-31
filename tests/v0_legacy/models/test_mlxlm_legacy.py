import warnings
from dataclasses import asdict

import pytest

try:
    import mlx.core as mx
    import mlx_lm  # noqa: F401

    assert mx.metal.is_available()
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    warnings.warn("Skipping MLXLM legacy tests because MLX is not available.")

from outlines import models, samplers, generate
from outlines.v0_legacy.generate.api import GeneratorV0Adapter
from outlines.v0_legacy.models.mlxlm import mlxlm_params_adapter

pytestmark = pytest.mark.skipif(
    not HAS_MLX,
    reason="MLX tests require Apple Silicon."
)


def test_mlxlm_legacy_init():
    # minimal
    with pytest.warns(
        DeprecationWarning,
        match="The `mlxlm` function is deprecated",
    ):
        model = models.mlxlm("mlx-community/SmolLM-135M-Instruct-4bit")
    assert isinstance(model, models.MLXLM)
    assert isinstance(model.model, mlx_lm.models.llama.Model)

    # with optional arguments
    with pytest.warns(
        DeprecationWarning,
        match="The `mlxlm` function is deprecated",
    ):
        model = models.mlxlm(
            "mlx-community/SmolLM-135M-Instruct-4bit",
            tokenizer_config={"max_length": 100},
            model_config={"max_length": 100},
            adapter_path=None,
            lazy=False,
        )
    assert isinstance(model, models.MLXLM)
    assert isinstance(model.model, mlx_lm.models.llama.Model)
    assert isinstance(model.mlx_tokenizer, mlx_lm.tokenizer_utils.TokenizerWrapper)


def test_mlxlm_legacy_params_adapter_generation_params():
    # default
    result = mlxlm_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result.get("stop") is None
    assert result.get("seed") is None
    assert result.get("max_tokens") == int(1e9)

    # max_tokens
    result = mlxlm_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": 2},
        {}
    )
    assert result.get("max_tokens") == 2

    # stop_at
    with pytest.raises(
        NotImplementedError,
        match="The `mlx-lm` library does not support `stop_at`."
    ):
        result = mlxlm_params_adapter(
            asdict(samplers.multinomial().sampling_params),
            {"stop_at": "foo", "seed": None, "max_tokens": None},
            {}
        )

    # seed
    with pytest.raises(
        NotImplementedError,
        match="The `mlx-lm` library does not support `seed`."
    ):
        result = mlxlm_params_adapter(
            asdict(samplers.multinomial().sampling_params),
            {"stop_at": None, "seed": 2, "max_tokens": None},
            {}
        )


def test_mlxlm_legacy_params_adapter_sampling_params():
    # beam search
    with pytest.raises(
        NotImplementedError,
        match="The `mlx-lm` library does not support Beam Search."
    ):
        result = mlxlm_params_adapter(
            asdict(samplers.beam_search().sampling_params),
            {"stop_at": None, "seed": None, "max_tokens": None},
            {}
        )

    # num_samples
    with pytest.raises(
        NotImplementedError,
        match="The `mlx-lm` library does not allow to take several samples."
    ):
        result = mlxlm_params_adapter(
            asdict(samplers.multinomial(samples=2).sampling_params),
            {"stop_at": None, "seed": None, "max_tokens": None},
            {}
        )

    # top_k
    with pytest.raises(
        NotImplementedError,
        match="The `mlx-lm` library does not support top_k."
    ):
        result = mlxlm_params_adapter(
            asdict(samplers.multinomial(top_k=2).sampling_params),
            {"stop_at": None, "seed": None, "max_tokens": None},
            {}
        )

    # temperature
    with pytest.warns(
        DeprecationWarning,
        match="The `temperature` parameter is not available",
    ):
        result = mlxlm_params_adapter(
            {
                "sampler": "multinomial",
                "num_samples": 1,
                "top_p": None,
                "top_k": None,
                "temperature": 0.5
            },
            {"stop_at": None, "seed": None, "max_tokens": None},
            {}
        )
        assert result.get("temperature") is None

    # top_p
    with pytest.warns(
        DeprecationWarning,
        match="The `top_p` parameter is not available",
    ):
        result = mlxlm_params_adapter(
            {
                "sampler": "multinomial",
                "num_samples": 1,
                "top_p": 2,
                "top_k": None,
                "temperature": None
            },
            {"stop_at": None, "seed": None, "max_tokens": None},
            {}
        )
        assert result.get("top_p") is None


def test_mlxlm_legacy_params_adapter_model_specific_params():
    result = mlxlm_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {"length_penalty": 0.5}
    )
    assert result.get("length_penalty") == 0.5


def test_mlxlm_legacy_call_generation():
    with pytest.warns(
        DeprecationWarning,
        match="The `mlxlm` function is deprecated",
    ):
        model = models.mlxlm("mlx-community/SmolLM-135M-Instruct-4bit")

    with pytest.warns(
        DeprecationWarning,
        match="The `text` function is deprecated",
    ):
        generator = generate.text(
            model,
            samplers.multinomial()
        )
    assert isinstance(generator, GeneratorV0Adapter)

    result = generator("Hello, world!", 10)
    assert isinstance(result, str)
