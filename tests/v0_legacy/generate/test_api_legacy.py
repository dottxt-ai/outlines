from io import BytesIO
from urllib.request import urlopen
from dataclasses import asdict

import pytest
import transformers
from PIL import Image
from anthropic import Anthropic as AnthropicClient

from outlines import models, samplers
from outlines.generator import SteerableGenerator
from outlines.models import TransformersVision  # type: ignore
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
            "erwanf/gpt2-mini",
            transformers.AutoModelForCausalLM
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
        adapter = GeneratorV0Adapter(model, None, samplers.greedy())

    # unknown output type
    with pytest.raises(
        TypeError,
        match="is currently not supported",
    ):
        adapter = GeneratorV0Adapter(
            transformers_model, type, samplers.greedy()
        )

    # valid initialization
    adapter = GeneratorV0Adapter(
        transformers_model,
        int,
        samplers.greedy(),
    )
    assert isinstance(adapter, GeneratorV0Adapter)
    assert adapter.model == transformers_model
    assert adapter.sampling_params == asdict(samplers.greedy().sampling_params)
    assert isinstance(adapter.generator, SteerableGenerator)
    assert adapter.generator.model == transformers_model
    assert isinstance(adapter.generator.logits_processor, RegexLogitsProcessor)


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

    generator = create_generator_v0_adapter(llama_cpp_model)
    result = generator.create_inference_params(**args, **kwargs)
    assert isinstance(result, dict)

    generator = create_generator_v0_adapter(transformers_vision_model)
    result = generator.create_inference_params(**args, **kwargs)
    assert isinstance(result, dict)

    generator = create_generator_v0_adapter(transformers_model)
    result = generator.create_inference_params(**args, **kwargs)
    assert isinstance(result, dict)


def test_generator_vision_v0_adapter_merge_prompts_and_media(transformers_vision_model):
    generator = create_generator_v0_adapter(transformers_vision_model)
    prompts = ["Hello, world!", "Goodbye, world!"]
    media = [
        img_from_url(
            "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
        ),
        img_from_url(
            "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
        )
    ]
    result = generator.merge_prompts_and_media(prompts, media)
    assert isinstance(result, dict)
    assert result["text"] == prompts
    assert result["images"] == media
