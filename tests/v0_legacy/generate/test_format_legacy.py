import pytest
from dataclasses import asdict

from transformers import AutoModelForCausalLM

from outlines import models, samplers
from outlines.generator import SteerableGenerator, Generator
from outlines.processors import RegexLogitsProcessor
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.generate.format import format


@pytest.fixture(scope="session")
def openai_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        return models.openai("gpt-4o")


@pytest.fixture(scope="session")
def transformers_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers` function is deprecated",
    ):
        return models.transformers("erwanf/gpt2-mini")


@pytest.fixture(scope="session")
def transformers_vision_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers_vision` function is deprecated",
    ):
        return models.transformers_vision(
            "erwanf/gpt2-mini",
            AutoModelForCausalLM
        )


def test_format_legacy_openai(openai_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `format` function is deprecated",
    ):
        with pytest.raises(
            NotImplementedError,
            match="Cannot use Python type-structured generation with an OpenAI model"
            + " due to the limitations of the OpenAI API.",
        ):
            format(openai_model, int)


def test_format_legacy_transformers_vision(transformers_vision_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `format` function is deprecated",
    ):
        generator = format(transformers_vision_model, int)
    assert isinstance(generator, GeneratorVisionV0Adapter)
    assert generator.model == transformers_vision_model
    assert generator.sampling_params == asdict(
        samplers.multinomial().sampling_params
    )
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_vision_model
    assert isinstance(
        generator.generator.logits_processor,
        RegexLogitsProcessor
    )


def test_format_legacy_standard_model(transformers_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `format` function is deprecated",
    ):
        generator = format(transformers_model, int, samplers.greedy())
    assert isinstance(generator, GeneratorV0Adapter)
    assert generator.model == transformers_model
    assert generator.sampling_params == asdict(
        samplers.greedy().sampling_params)
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_model
    assert isinstance(
        generator.generator.logits_processor,
        RegexLogitsProcessor
    )
