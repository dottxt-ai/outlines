import pytest
from typing import AsyncGenerator, Generator as TypingGenerator, Literal

import transformers
from outlines_core import Index, Vocabulary

import outlines
from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor
from outlines.generator import (
    BlackBoxGenerator,
    SteerableGenerator,
    Generator,
    AsyncBlackBoxGenerator,
)
from outlines.models import AsyncVLLM, VLLM
from outlines.processors import (
    OutlinesLogitsProcessor,
)
from outlines.types import CFG
from tests.test_utils.mock_openai_client import (
    MockAsyncOpenAIClient,
    MockOpenAIClient,
)


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


# We used the mocked vllm model to test the black box generator
async_openai_client = MockAsyncOpenAIClient()
openai_client = MockOpenAIClient()
mock_responses = [
    (
        {
            'messages': [
                {'role': "user", 'content': 'Write a very short sentence'}
            ],
            'model': MODEL_NAME,
            'max_tokens': 10,
            'extra_body': {'guided_regex': '("[^"]*")'},
        },
        "Mock response"
    ),
    (
        {
            'messages': [
                {'role': "user", 'content': 'Write a very short sentence'}
            ],
            'model': MODEL_NAME,
            'max_tokens': 10,
            'extra_body': {'guided_regex': '("[^"]*")'},
            'stream': True,
        },
        ["Mock", "response"]
    ),
]
async_openai_client.add_mock_responses(mock_responses)
openai_client.add_mock_responses(mock_responses)


@pytest.fixture(scope="session")
def steerable_model():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )
    return model


@pytest.fixture(scope="session")
def sample_processor():
    vocabulary = Vocabulary.from_pretrained("openai-community/gpt2")
    index = Index(r"[0-9]{3}", vocabulary)
    return OutlinesCoreLogitsProcessor(index, "torch")


@pytest.fixture(scope="module")
def black_box_sync_model():
    return VLLM(openai_client, MODEL_NAME)


@pytest.fixture(scope="module")
def black_box_async_model():
    return AsyncVLLM(async_openai_client, MODEL_NAME)


# SteerableGenerator


def test_steerable_generator_init_valid_processor(steerable_model, sample_processor):
    generator = SteerableGenerator.from_processor(steerable_model, sample_processor)
    assert generator.logits_processor == sample_processor
    assert generator.model == steerable_model


def test_steerable_generator_init_cfg_output_type(steerable_model):
    generator = SteerableGenerator(steerable_model, CFG('start: "a"'))
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


def test_black_box_generator_init(black_box_sync_model):
    generator = BlackBoxGenerator(black_box_sync_model, Literal["foo", "bar"])
    assert generator.model == black_box_sync_model
    assert generator.output_type == Literal["foo", "bar"]

def test_black_box_generator_call(black_box_sync_model):
    generator = BlackBoxGenerator(black_box_sync_model, str)
    result = generator("Write a very short sentence", max_tokens=10)
    assert isinstance(result, str)


def test_black_box_generator_stream(black_box_sync_model):
    generator = BlackBoxGenerator(black_box_sync_model, str)
    result = generator.stream("Write a very short sentence", max_tokens=10)
    assert isinstance(result, TypingGenerator)
    assert isinstance(next(result), str)


# AsyncBlackBoxGenerator


def test_async_black_box_generator_init(black_box_async_model):
    generator = AsyncBlackBoxGenerator(black_box_async_model, Literal["foo", "bar"])
    assert generator.model == black_box_async_model
    assert generator.output_type == Literal["foo", "bar"]


@pytest.mark.asyncio
async def test_async_black_box_generator_call(black_box_async_model):
    generator = AsyncBlackBoxGenerator(black_box_async_model, str)
    result = await generator("Write a very short sentence", max_tokens=10)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_async_black_box_generator_stream(black_box_async_model):
    generator = AsyncBlackBoxGenerator(black_box_async_model, str)
    result = generator.stream("Write a very short sentence", max_tokens=10)
    assert isinstance(result, AsyncGenerator)
    async for chunk in result:
        assert isinstance(chunk, str)
        break  # Just check the first chunk


# Generator


def test_generator_init_no_model():
    with pytest.raises(ValueError):
        Generator(None, Literal["foo", "bar"])


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


def test_generator_black_box_sync_output_type(black_box_sync_model):
    generator = Generator(black_box_sync_model, Literal["foo", "bar"])
    assert isinstance(generator, BlackBoxGenerator)
    assert generator.model == black_box_sync_model
    assert generator.output_type == Literal["foo", "bar"]


def test_generator_black_box_sync_processor(black_box_sync_model, sample_processor):
    with pytest.raises(NotImplementedError):
        Generator(black_box_sync_model, processor=sample_processor)


def test_generator_black_box_async_output_type(black_box_async_model):
    generator = Generator(black_box_async_model, Literal["foo", "bar"])
    assert isinstance(generator, AsyncBlackBoxGenerator)
    assert generator.model == black_box_async_model
    assert generator.output_type == Literal["foo", "bar"]


def test_generator_black_box_async_processor(black_box_async_model, sample_processor):
    with pytest.raises(NotImplementedError):
        Generator(black_box_async_model, processor=sample_processor)
