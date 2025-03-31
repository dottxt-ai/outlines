from dataclasses import asdict

import interegular
import pytest
from transformers import AutoModelForCausalLM

from outlines import models, samplers
from outlines.generator import SteerableGenerator, Generator
from outlines.processors import GuideLogitsProcessor
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.generate.fsm import fsm


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


def test_fsm_legacy_openai(openai_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `fsm` function is deprecated",
    ):
        with pytest.raises(
            NotImplementedError,
            match="Cannot use FSM-based generation with an OpenAI model"
            + "due to the limitations of the OpenAI API.",
        ):
            fsm(
                openai_model,
                interegular.parse_pattern(r"[a-z]+").to_fsm()
            )


def test_fsm_legacy_transformers_vision(transformers_vision_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `fsm` function is deprecated",
    ):
        generator = fsm(
            transformers_vision_model,
            interegular.parse_pattern(r"[a-z]+").to_fsm()
        )
    assert isinstance(generator, GeneratorVisionV0Adapter)
    assert generator.model == transformers_vision_model
    assert generator.sampling_params == asdict(
        samplers.multinomial().sampling_params
    )
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_vision_model
    assert isinstance(
        generator.generator.logits_processor,
        GuideLogitsProcessor
    )


def test_fsm_legacy_standard_model(transformers_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `fsm` function is deprecated",
    ):
        generator = fsm(
            transformers_model,
            interegular.parse_pattern(r"[a-z]+").to_fsm(),
            samplers.greedy()
        )
    assert isinstance(generator, GeneratorV0Adapter)
    assert generator.model == transformers_model
    assert generator.sampling_params == asdict(
        samplers.greedy().sampling_params)
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_model
    assert isinstance(
        generator.generator.logits_processor,
        GuideLogitsProcessor
    )
