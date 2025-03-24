import os
import pytest
from typing import Literal, Generator as TypingGenerator

import interegular
import transformers
from openai import OpenAI

import outlines
from outlines.generator import BlackBoxGenerator, SteerableGenerator, Generator
from outlines.models import BlackBoxModel, SteerableModel
from outlines.processors import (
    OutlinesLogitsProcessor,
    RegexLogitsProcessor,
)
from outlines.types import CFG, FSM


@pytest.fixture(scope="session")
def steerable_model():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )
    return model


@pytest.fixture(scope="session")
def sample_processor():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )
    processor = RegexLogitsProcessor(
        regex_string="[0-9]{3}",
        tokenizer=model.tokenizer,
        tensor_library_name=model.tensor_library_name,
    )
    return processor


@pytest.fixture(scope="session")
def black_box_model():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "MOCK_API_KEY"
    client = OpenAI(api_key=api_key)
    model = outlines.from_openai(client, model_name="gpt-4o-mini")
    return model


# SteerableGenerator


def test_steerable_generator_init_valid_processor(steerable_model, sample_processor):
    generator = SteerableGenerator.from_processor(steerable_model, sample_processor)
    assert generator.logits_processor == sample_processor
    assert generator.model == steerable_model


def test_steerable_generator_init_invalid_processor(steerable_model):
    with pytest.raises(TypeError):
        SteerableGenerator.from_processor(steerable_model, Literal["foo", "bar"])


def test_steerable_generator_init_cfg_output_type(steerable_model):
    generator = SteerableGenerator(steerable_model, CFG('start: "a"'))
    assert generator.model == steerable_model
    assert isinstance(generator.logits_processor, OutlinesLogitsProcessor)


def test_steerable_generator_init_fsm_output_type(steerable_model):
    generator = SteerableGenerator(
        steerable_model,
        FSM(interegular.parse_pattern(r"abc").to_fsm())
    )
    assert generator.model == steerable_model
    assert isinstance(generator.logits_processor, OutlinesLogitsProcessor)


def test_steerable_generator_init_other_output_type(steerable_model):
    generator = SteerableGenerator(steerable_model, Literal["foo", "bar"])
    assert generator.model == steerable_model
    assert isinstance(generator.logits_processor, OutlinesLogitsProcessor)


def test_steerable_generator_init_invalid_output_type(steerable_model, sample_processor):
    with pytest.raises(ValueError):
        SteerableGenerator(steerable_model, sample_processor)


def test_steerable_generator_call(steerable_model):
    generator = SteerableGenerator(steerable_model, Literal["foo", "bar"])
    result = generator("foo", max_new_tokens=10)
    assert isinstance(result, str)


def test_steerable_generator_stream(steerable_model):
    with pytest.raises(NotImplementedError):
        generator = SteerableGenerator(steerable_model, Literal["foo", "bar"])
        result = generator.stream("foo", max_tokens=10)
        assert isinstance(result, TypingGenerator)
        assert isinstance(next(result), str)


# BlackBoxGenerator


def test_black_box_generator_init_valid(black_box_model):
    generator = BlackBoxGenerator(black_box_model, Literal["foo", "bar"])
    assert generator.model == black_box_model
    assert generator.output_type == Literal["foo", "bar"]


def test_black_box_generator_init_invalid(black_box_model):
    with pytest.raises(NotImplementedError):
        BlackBoxGenerator(black_box_model, CFG("foo"))

    with pytest.raises(NotImplementedError):
        BlackBoxGenerator(black_box_model, FSM("foo"))


@pytest.mark.api_call
def test_black_box_generator_call(black_box_model):
    generator = BlackBoxGenerator(black_box_model, dict)
    result = generator("write a very short json object", max_tokens=10)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_black_box_generator_stream(black_box_model):
    generator = BlackBoxGenerator(black_box_model, dict)
    result = generator.stream("write a very short json object", max_tokens=10)
    assert isinstance(result, TypingGenerator)
    assert isinstance(next(result), str)


# Generator


def test_generator_init_multiple_output_type(steerable_model, sample_processor):
    with pytest.raises(ValueError):
        Generator(steerable_model, Literal["foo", "bar"], processor=sample_processor)


def test_generator_steerable_output_type(steerable_model):
    generator = Generator(steerable_model, Literal["foo", "bar"])
    assert isinstance(generator, SteerableGenerator)
    assert generator.model == steerable_model
    assert isinstance(generator.logits_processor, OutlinesLogitsProcessor)


def test_generator_steerable_processor(steerable_model, sample_processor):
    generator = Generator(steerable_model, processor=sample_processor)
    assert isinstance(generator, SteerableGenerator)
    assert generator.model == steerable_model
    assert isinstance(generator.logits_processor, OutlinesLogitsProcessor)


def test_generator_black_box_output_type(black_box_model):
    generator = Generator(black_box_model, Literal["foo", "bar"])
    assert isinstance(generator, BlackBoxGenerator)
    assert generator.model == black_box_model
    assert generator.output_type == Literal["foo", "bar"]


def test_generator_black_box_processor(black_box_model, sample_processor):
    with pytest.raises(NotImplementedError):
        Generator(black_box_model, processor=sample_processor)
