import re
from enum import Enum

import pytest
from pydantic import BaseModel

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

import outlines
from outlines.models.vllm_offline import (
    VLLMOffline,
    VLLMOfflineTypeAdapter,
    from_vllm_offline
)
from outlines.types import Regex


TEST_MODEL = "erwanf/gpt2-mini"

pytestmark = pytest.mark.skipif(
    not HAS_VLLM,
    reason="vLLM models can only be run on GPU."
)


def test_vllm_model_initialization():
    model = from_vllm_offline(LLM(TEST_MODEL))
    assert isinstance(model, VLLMOffline)
    assert isinstance(model.model, LLM)
    assert isinstance(model.type_adapter, VLLMOfflineTypeAdapter)


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    model = outlines.from_vllm_offline(LLM(TEST_MODEL))
    return model


def test_vllm_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_vllm_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


def test_vllm_inference_kwargs(model):
    result = model("Write a short story about a cat.", sampling_params=SamplingParams(max_tokens=2), use_tqdm=True)
    assert isinstance(result, str)
    assert len(result) <= 20


def test_vllm_invalid_inference_kwargs(model):
    with pytest.raises(TypeError):
        model("Respond with one word. Not more.", foo="bar")


def test_vllm_regex(model):
    result = model("Give a number between 0 and 9.", Regex(r"[0-9]"))
    assert isinstance(result, str)
    assert re.match(r"[0-9]", result)


def test_vllm_json(model):
    class Character(BaseModel):
        name: str

    result = model("Create a character with a name.", Character)
    assert "name" in result


def test_vllm_choice(model):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model("Cat or dog?", Foo)
    assert result in ["cat", "dog"]


def test_vllm_batch_samples(model):
    result = model(
        "Respond with one word. Not more.",
        sampling_params=SamplingParams(n=2)
    )
    assert isinstance(result, list)
    assert len(result) == 2

    result = model(
        ["Respond with one word. Not more.", "Respond with one word. Not more."]
    )
    assert isinstance(result, list)
    assert len(result) == 2

    result = model(
        ["Respond with one word. Not more.", "Respond with one word. Not more."],
        sampling_params=SamplingParams(n=2)
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2


def test_vllm_streaming(model):
    with pytest.raises(
        NotImplementedError,
        match="Streaming is not available for the vLLM integration."
    ):
        model.stream("Respond with one word. Not more.")
