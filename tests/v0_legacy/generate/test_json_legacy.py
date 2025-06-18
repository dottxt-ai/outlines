import json as pyjson
import pytest
from dataclasses import asdict

from pydantic import BaseModel
from transformers import AutoModelForCausalLM

from outlines import models, samplers
from outlines.generator import SteerableGenerator, Generator
from outlines.processors import RegexLogitsProcessor
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.generate.json import json


JSON_SCHEMA_STRING = """
{
    "type": "object",
    "properties": {
        "foo": {
            "type": "string"
        }
    },
    "required": ["foo"]
}
"""


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


def test_json_legacy_standard_callable(transformers_model):
    def foo(bar: str) -> str:
        return bar
    with pytest.warns(
        DeprecationWarning,
        match="The `json` function is deprecated",
    ):
        generator = json(transformers_model, foo, samplers.greedy())
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


def test_json_legacy_standard_json_string(transformers_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `json` function is deprecated",
    ):
        generator = json(transformers_model, JSON_SCHEMA_STRING)
    assert isinstance(generator, GeneratorV0Adapter)
    assert generator.model == transformers_model
    assert generator.sampling_params == asdict(
        samplers.multinomial().sampling_params
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


def test_json_legacy_standard_pydantic(transformers_model):
    class Foo(BaseModel):
        foo: str
    with pytest.warns(
        DeprecationWarning,
        match="The `json` function is deprecated",
    ):
        generator = json(transformers_model, Foo)
    assert isinstance(generator, GeneratorV0Adapter)
    assert generator.model == transformers_model
    assert generator.sampling_params == asdict(
        samplers.multinomial().sampling_params
    )
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_model
    assert (
        generator.format_sequence('{"foo": "value"}')
        == Foo.model_validate_json('{"foo": "value"}')
    )
    assert isinstance(
        generator.generator.logits_processor, RegexLogitsProcessor
    )


def test_json_legacy_transformers_vision(transformers_vision_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `json` function is deprecated",
    ):
        generator = json(transformers_vision_model, JSON_SCHEMA_STRING)
    assert isinstance(generator, GeneratorVisionV0Adapter)
    assert generator.model == transformers_vision_model
    assert generator.sampling_params == asdict(
        samplers.multinomial().sampling_params
    )
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_vision_model
    assert (
        generator.format_sequence('{"key": "value"}')
        == pyjson.loads('{"key": "value"}')
    )
    assert isinstance(
        generator.generator.logits_processor, RegexLogitsProcessor
    )


def test_json_legacy_openai_wrong_sampler(openai_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `json` function is deprecated",
    ):
        with pytest.raises(
            NotImplementedError,
            match="The OpenAI API does not support any other sampling "
            + "algorithm than the multinomial sampler.",
        ):
            json(openai_model, JSON_SCHEMA_STRING, samplers.greedy())


def test_json_legacy_openai_pydantic(openai_model):
    class Foo(BaseModel):
        foo: str
    with pytest.warns(
        DeprecationWarning,
        match="The `json` function is deprecated",
    ):
        generator = json(openai_model, Foo)
    assert isinstance(generator, models.OpenAI)
    assert (
        generator.format_sequence('{"foo": "value"}')
        == Foo.model_validate_json('{"foo": "value"}')
    )


def test_json_legacy_openai_json_string(openai_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `json` function is deprecated",
    ):
        generator = json(openai_model, JSON_SCHEMA_STRING)
    assert isinstance(generator, models.OpenAI)
    assert (
        generator.format_sequence('{"key": "value"}')
        == pyjson.loads('{"key": "value"}')
    )


def test_json_legacy_openai_not_supported(openai_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `json` function is deprecated",
    ):
        with pytest.raises(ValueError, match="Cannot parse schema"):
            json(openai_model, int)
