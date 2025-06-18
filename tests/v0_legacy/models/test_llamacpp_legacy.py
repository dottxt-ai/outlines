import warnings
from dataclasses import asdict

import pytest
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer, LlamaTokenizer # type: ignore

from outlines import models, samplers, generate
from outlines.v0_legacy.models.llamacpp import llamacpp_params_adapter
from outlines.v0_legacy.generate.api import GeneratorV0Adapter


def test_llamacpp_legacy_init():
    # without tokenizer
    with pytest.warns(
        DeprecationWarning,
        match="The `llamacpp` function is deprecated",
    ):
        with pytest.warns(
            match="The pre-tokenizer in `llama.cpp` handles unicode improperly"
        ):
            model = models.llamacpp(
                repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
                filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
            )
    assert isinstance(model, models.LlamaCpp)
    assert isinstance(model.model, Llama)
    assert not model.model.verbose

    # with tokenizer
    with pytest.warns(
        DeprecationWarning,
        match="The `llamacpp` function is deprecated",
    ):
        model = models.llamacpp(
            repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
            tokenizer=LlamaHFTokenizer.from_pretrained("distilbert-base-uncased"),
            n_ctx=0,
            verbose=True,
        )
    assert isinstance(model, models.LlamaCpp)
    assert isinstance(model.model, Llama)
    assert model.model.verbose
    assert isinstance(model.model.tokenizer(), LlamaTokenizer)


def test_llamacpp_legacy_params_adapter_generation_params():
    result = llamacpp_params_adapter(
         asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert not result.get("stop")
    assert not result.get("seed")
    assert result.get("max_tokens") == -1

    result = llamacpp_params_adapter(
         asdict(samplers.multinomial().sampling_params),
        {"stop_at": "foo", "seed": 2, "max_tokens": 10},
        {}
    )
    assert result.get("stop") == "foo"
    assert result.get("seed") == 2
    assert result.get("max_tokens") == 10


def test_llamacpp_legacy_params_adapter_sampling_params():
    with pytest.raises(
        NotImplementedError,
        match="does not support Beam Search"
    ):
        result = llamacpp_params_adapter(
            asdict(samplers.beam_search().sampling_params),
            {"stop_at": None, "seed": None, "max_tokens": None},
            {}
        )

    with pytest.raises(
        NotImplementedError,
        match="does not allow to take several samples"
    ):
        result = llamacpp_params_adapter(
            asdict(samplers.multinomial(samples=2).sampling_params),
            {"stop_at": None, "seed": None, "max_tokens": None},
            {}
        )

    result = llamacpp_params_adapter(
        {"sampler": "multinomial", "num_samples": 1, "top_p": None, "top_k": None, "temperature": None},
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result.get("top_p") == 1.0
    assert result.get("top_k") == -1
    assert result.get("temperature") == 1.0

    result = llamacpp_params_adapter(
        {"sampler": "greedy", "num_samples": 1, "top_p": 2, "top_k": 0, "temperature": 0.5},
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result.get("top_p") == 2
    assert result.get("top_k") == 0
    assert result.get("temperature") == 0.5


def test_llamacpp_legacy_params_adapter_model_specific_params():
    result = llamacpp_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert result.get("min_p") == 0.0
    assert result.get("repeat_penalty") == 1.0

    result = llamacpp_params_adapter(
         asdict(samplers.multinomial().sampling_params),
        {"stop_at": "foo", "seed": 2, "max_tokens": 10},
        {"min_p": 0.5, "repeat_penalty": 1.5, "stream": True}
    )
    assert result.get("min_p") == 0.5
    assert result.get("repeat_penalty") == 1.5
    assert not hasattr(result, "stream")


def test_llamacpp_legacy_call_generation():
    with pytest.warns(
        DeprecationWarning,
        match="The `llamacpp` function is deprecated",
    ):
        model = models.llamacpp(
            repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
        )
    with pytest.warns(
        DeprecationWarning,
        match="The `text` function is deprecated",
    ):
        generator = generate.text(
            model,
            samplers.multinomial(
                top_p=0.9,
                top_k=10,
                temperature=0.7
            )
        )
    assert isinstance(generator, GeneratorV0Adapter)

    result = generator(
        "Hello, world!",
        10,
        "foo",
        2,
        min_p=0.5,
        repeat_penalty=1.5,
    )
    assert isinstance(result, str)
    assert isinstance(generator, GeneratorV0Adapter)

    result = generator(
        "Hello, world!",
        10,
        "foo",
        2,
        min_p=0.5,
        repeat_penalty=1.5,
    )
    assert isinstance(result, str)
