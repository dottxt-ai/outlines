import warnings
from dataclasses import asdict

import pytest

try:
    from vllm import LLM
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    warnings.warn("Skipping vLLM legacy tests because vLLM is not available.")

from outlines import generate, models, samplers
from outlines.v0_legacy.generate.api import GeneratorV0Adapter
from outlines.v0_legacy.models.vllm_offline import vllm_offline_params_adapter

pytestmark = pytest.mark.skipif(
    not HAS_VLLM,
    reason="vLLM models can only be run on GPU."
)


@pytest.fixture
def model():
    with pytest.warns(
        DeprecationWarning,
        match="The `vllm` function is deprecated",
    ):
        model_instance = models.vllm(
            "erwanf/gpt2-mini",
            seed=1,
        )
    return model_instance


def test_vllm_legacy_init(model):
    assert isinstance(model, models.VLLM)
    assert isinstance(model.model, LLM)


def test_vllm_legacy_params_adapter_generation_params():
    # default
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].stop_strings is None
    assert result["sampling_params"].seed is None
    assert result["sampling_params"].max_tokens is None

    # max_tokens
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": 2},
        {}
    )
    assert result["sampling_params"].max_tokens == 2

    # seed
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": 2, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].seed == 2

    # stop_at str
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": "foo", "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].stop_strings == ["foo"]

    # stop_at list
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": ["foo", "bar"], "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].stop_strings == ["foo", "bar"]


def test_vllm_legacy_params_adapter_sampling_params():
    # default
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].n == 1
    assert result["sampling_params"].top_p == 1.0
    assert result["sampling_params"].top_k == -1
    assert result["sampling_params"].temperature == 1.0

    # beam search
    result = vllm_offline_params_adapter(
        asdict(samplers.beam_search().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].use_beam_search is True

    # num_samples
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial(samples=2).sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].n == 2
    assert result["sampling_params"].best_of == 2

    # top_p
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial(top_p=0.5).sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].top_p == 0.5

    # top_k
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial(top_k=2).sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].top_k == 2

    # temperature
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial(temperature=0.5).sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result["sampling_params"].temperature == 0.5


def test_vllm_legacy_params_adapter_model_specific_params():
    result = vllm_offline_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {"use_tqdm": False}
    )
    assert not result.get("use_tqdm")


def test_vllm_legacy_call_generation(model):
    # default
    with pytest.warns(
        DeprecationWarning,
        match="The `text` function is deprecated",
    ):
        generator = generate.text(
            model,
            samplers.multinomial()
        )
    assert isinstance(generator, GeneratorV0Adapter)

    result = generator("Hello, world!")
    assert isinstance(result, str)

    # with args
    with pytest.warns(
        DeprecationWarning,
        match="The `text` function is deprecated",
    ):
        generator = generate.text(
            model,
            samplers.multinomial(
                samples=2, top_p=0.5, top_k=2, temperature=0.5
            )
        )
    assert isinstance(generator, GeneratorV0Adapter)

    result = generator(
        "Hello, world!", 10, stop_at="foo", seed=1, use_tqdm=False
    )
    assert isinstance(result, list)
    for completion in result:
        assert isinstance(completion, str)
