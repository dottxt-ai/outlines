from io import BytesIO
from dataclasses import asdict
from typing import Generator
from urllib.request import urlopen

import pytest
import transformers
from PIL import Image
from anthropic import Anthropic as AnthropicClient

from outlines import models, samplers
from outlines.generator import SteerableGenerator
from outlines.v0_legacy.models.transformers_vision import TransformersVision
from outlines.processors import RegexLogitsProcessor
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)


def img_from_url(url):
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")


@pytest.fixture(scope="session")
def transformers_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers` function is deprecated",
    ):
        return models.transformers(
            "erwanf/gpt2-mini"
        )


@pytest.fixture(scope="session")
def transformers_vision_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers_vision` function is deprecated",
    ):
        return models.transformers_vision(
            "trl-internal-testing/tiny-LlavaForConditionalGeneration",
            model_class=transformers.LlavaForConditionalGeneration,
            processor_class=transformers.AutoProcessor,
        )


@pytest.fixture(scope="session")
def llama_cpp_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `llamacpp` function is deprecated",
    ):
        return models.llamacpp(
            "M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            "TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
        )


def create_generator_v0_adapter(model):
    if isinstance(model, TransformersVision):
        return GeneratorVisionV0Adapter(
            model,
            int,
            samplers.greedy(),
        )
    else:
        return GeneratorV0Adapter(
            model,
            int,
            samplers.greedy(),
        )


def test_generator_v0_adapter_init(transformers_model):
    # wrong model type
    model = models.from_anthropic(AnthropicClient(), "claude-3-haiku-20240307")
    with pytest.raises(
        ValueError,
        match="You can only use the v0 API with models that were already"
        + "available in v0. Got <class 'outlines.models.anthropic.Anthropic'>.",
    ):
        GeneratorV0Adapter(model, None, samplers.greedy())

    # unknown output type
    with pytest.raises(
        TypeError,
        match="is currently not supported",
    ):
        GeneratorV0Adapter(
            transformers_model, type, samplers.greedy()
        )

    # valid initialization
    generator_v0_adapter = GeneratorV0Adapter(
        transformers_model,
        int,
        samplers.greedy(),
    )
    assert isinstance(generator_v0_adapter, GeneratorV0Adapter)
    assert generator_v0_adapter.model == transformers_model
    assert generator_v0_adapter.sampling_params == asdict(samplers.greedy().sampling_params)
    assert isinstance(generator_v0_adapter.generator, SteerableGenerator)
    assert generator_v0_adapter.generator.model == transformers_model
    assert isinstance(generator_v0_adapter.generator.logits_processor, RegexLogitsProcessor)


def test_generator_v0_adapter_create_inference_params(
    llama_cpp_model,
    transformers_vision_model,
    transformers_model,
):
    args = {
        "max_tokens": 10,
        "stop_at": "\n",
        "seed": 42
    }
    kwargs = {
        "foo": "bar",
    }

    result = (
        create_generator_v0_adapter(llama_cpp_model)
        .create_inference_params(**args, **kwargs)
    )
    assert isinstance(result, dict)

    result = (
        create_generator_v0_adapter(transformers_vision_model)
        .create_inference_params(**args, **kwargs)
    )
    assert isinstance(result, dict)

    result = (
        create_generator_v0_adapter(transformers_model)
        .create_inference_params(**args, **kwargs)
    )
    assert isinstance(result, dict)


def test_generator_vision_v0_adapter_merge_prompts_and_media(transformers_vision_model):
    generator_v0_adapter = create_generator_v0_adapter(transformers_vision_model)
    prompts = ["Hello, world!", "Goodbye, world!"]
    media = [
        img_from_url(
            "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
        ),
        img_from_url(
            "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
        )
    ]
    result = generator_v0_adapter.merge_prompts_and_media(prompts, media)
    assert isinstance(result, dict)
    assert result["text"] == prompts
    assert result["images"] == media


def test_generator_v0_adapter_call_stream(llama_cpp_model, transformers_model):
    generator_v0_adapter = create_generator_v0_adapter(transformers_model)
    result = generator_v0_adapter("Hello, world!", max_tokens=10)
    assert isinstance(result, str)
    result = generator_v0_adapter(["Hello, world!", "Goodbye, world!"], max_tokens=10)
    assert isinstance(result, list)
    for r in result:
        assert isinstance(r, str)

    generator_v0_adapter = create_generator_v0_adapter(llama_cpp_model)
    gen = generator_v0_adapter.stream("Hello, world!", max_tokens=10)
    assert isinstance(gen, Generator)


def test_generator_v0_adapter_vision_call_stream(transformers_vision_model):
    generator_v0_adapter = create_generator_v0_adapter(transformers_vision_model)
    media = img_from_url(
        "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
    )
    result = generator_v0_adapter("Hello, world!<image>", media, max_tokens=10)
    assert isinstance(result, str)
    with pytest.raises(
        NotImplementedError,
        match="Streaming is not implemented for Transformers models",
    ):
        gen = generator_v0_adapter.stream("Hello, world!<image>", media, max_tokens=10)
        assert isinstance(gen, Generator)
