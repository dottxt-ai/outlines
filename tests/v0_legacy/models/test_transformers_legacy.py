from dataclasses import asdict

import pytest
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    BertForSequenceClassification,
    BertTokenizer,
    MambaForCausalLM,
)

from outlines import models, samplers, generate
from outlines.v0_legacy.models.transformers import transformers_params_adapter
from outlines.v0_legacy.generate.api import GeneratorV0Adapter


def test_transformers_legacy_init():
    # Test with default model and tokenizer classes
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers` function is deprecated",
    ):
        model = models.transformers(
            model_name="erwanf/gpt2-mini",
        )
    assert isinstance(model, models.Transformers)
    assert isinstance(model.model, GPT2LMHeadModel)
    assert isinstance(model.tokenizer.tokenizer, GPT2TokenizerFast)

    # Test with non default model and tokenizer classes
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers` function is deprecated",
    ):
        model = models.transformers(
            model_name="bert-base-uncased",
            model_class=BertForSequenceClassification,
            tokenizer_class=BertTokenizer,
        )
    assert isinstance(model, models.Transformers)
    assert isinstance(model.model, BertForSequenceClassification)
    assert isinstance(model.tokenizer.tokenizer, BertTokenizer)

    # Test with model and tokenizer kwargs and device
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers` function is deprecated",
    ):
        model = models.transformers(
            model_name="erwanf/gpt2-mini",
            model_kwargs={"use_cache": False},
            tokenizer_kwargs={"model_max_length": 512},
            device="cpu",
        )
    assert isinstance(model, models.Transformers)
    assert isinstance(model.model, GPT2LMHeadModel)
    assert isinstance(model.tokenizer.tokenizer, GPT2TokenizerFast)
    assert model.tokenizer.tokenizer.model_max_length == 512
    assert not model.model.config.use_cache
    assert model.model.device.type == "cpu"


def test_mamba_legacy_init():
    with pytest.warns(
        DeprecationWarning,
        match="The `mamba` function is deprecated",
    ):
        model = models.mamba(
            model_name="erwanf/gpt2-mini",
        )
    assert isinstance(model, models.Transformers)
    assert isinstance(model.model, MambaForCausalLM)
    assert isinstance(model.tokenizer.tokenizer, GPT2TokenizerFast)


def test_transformers_legacy_params_adapter_generation_params():
    result = transformers_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {}
    )
    assert not result.get("stop")
    assert not result.get("seed")
    assert result.get("max_new_tokens") == 2**30

    result = transformers_params_adapter(
         asdict(samplers.multinomial().sampling_params),
        {"stop_at": "foo", "seed": 2, "max_tokens": 10},
        {}
    )
    assert result.get("stop_strings") == "foo"
    assert not result.get("seed")
    assert result.get("max_new_tokens") == 10


def test_transformers_legacy_params_adapter_sampling_params():
    result = transformers_params_adapter(
        {
            "sampler": "multinomial",
            "num_samples": None,
            "top_p": None,
            "top_k": None,
            "temperature": None,
        },
        {"stop_at": None, "seed": None, "max_tokens": None},
        {},
    )
    assert result.get("do_sample")
    assert result.get("num_return_sequences") == 1
    assert result.get("top_p") is None
    assert result.get("top_k") is None
    assert result.get("temperature") is None

    result = transformers_params_adapter(
        {
            "sampler": "beam_search",
            "num_samples": 2,
            "top_p": 2,
            "top_k": 0,
            "temperature": 0.5,
        },
        {"stop_at": None, "seed": None, "max_tokens": None},
        {},
    )
    assert result.get("num_beams") == 2
    assert result.get("num_return_sequences") == 2
    assert result.get("top_p") == 2
    assert result.get("top_k") == 0
    assert result.get("temperature") == 0.5


def test_transformers_legacy_params_adapter_model_specific_params():
    result = transformers_params_adapter(
        asdict(samplers.multinomial().sampling_params),
        {"stop_at": None, "seed": None, "max_tokens": None},
        {"length_penalty": 1.0}
    )
    assert result.get("length_penalty") == 1
    assert not result.get("foo")


def test_transformers_legacy_call_generation():
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers` function is deprecated",
    ):
        model = models.transformers(
            model_name="erwanf/gpt2-mini",
        )
    with pytest.warns(
        DeprecationWarning,
        match="The `text` function is deprecated",
    ):
        generator = generate.text(
            model,
            samplers.multinomial(
                samples=1,
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
        length_penalty=0.5,
    )
    assert isinstance(result, str)
