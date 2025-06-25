import json as pyjson
from dataclasses import asdict
from enum import Enum
from typing import Callable

import pytest
from pydantic import BaseModel
from transformers import AutoModelForCausalLM

from outlines import models, samplers
from outlines.generator import SteerableGenerator, Generator
from outlines.processors import RegexLogitsProcessor
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.generate.choice import choice


class EnumChoice(Enum):
    foo = "foo"
    bar = "bar"


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


def test_choice_legacy_standard_enum(transformers_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `choice` function is deprecated",
    ):
        generator = choice(transformers_model, EnumChoice, samplers.greedy())
    assert isinstance(generator, GeneratorV0Adapter)
    assert generator.model == transformers_model
    assert generator.sampling_params == asdict(
        samplers.greedy().sampling_params
    )
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_model
    assert (
        generator.format_sequence('{"key": "value"}')
        == pyjson.loads('{"key": "value"}')
    )
    assert isinstance(
        generator.generator.logits_processor, RegexLogitsProcessor
    )


def test_choice_legacy_standard_list_strings(transformers_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `choice` function is deprecated",
    ):
        generator = choice(transformers_model, ["foo", "bar"])
    assert isinstance(generator, GeneratorV0Adapter)
    assert generator.model == transformers_model
    assert generator.sampling_params == asdict(
        samplers.multinomial().sampling_params
    )
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_model
    assert generator.format_sequence("foo") == "foo"
    assert isinstance(
        generator.generator.logits_processor, RegexLogitsProcessor
    )


def test_choice_legacy_openai(openai_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `choice` function is deprecated",
    ):
        with pytest.raises(
            ValueError,
            match="The `choice` function with OpenAI only supports a list of "
            + "strings as choices.",
        ):
            choice(openai_model, EnumChoice)

        generator = choice(openai_model, ["foo", "bar"])
        assert isinstance(generator, Callable)
