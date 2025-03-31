from io import BytesIO
from urllib.request import urlopen

import pytest
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlamaTokenizer,
    LlamaTokenizerFast,
    LlavaProcessor,
)

from outlines import models, generate, samplers
from outlines.v0_legacy.generate.api import GeneratorV0Adapter


def img_from_url(url):
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")


def test_transformers_vision_legacy_init():
    # Test with only model name and model class
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers_vision` function is deprecated",
    ):
        model = models.transformers_vision(
            model_name="trl-internal-testing/tiny-LlavaForConditionalGeneration",
            model_class=LlavaForConditionalGeneration,
        )
    assert isinstance(model, models.TransformersVision)
    assert isinstance(model.model, LlavaForConditionalGeneration)
    assert isinstance(model.tokenizer.tokenizer, LlamaTokenizerFast)
    assert isinstance(model.processor, LlavaProcessor)

    # Test with model, tokenizer, and processor classes
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers_vision` function is deprecated",
    ):
        model = models.transformers_vision(
            model_name="trl-internal-testing/tiny-LlavaForConditionalGeneration",
            model_class=LlavaForConditionalGeneration,
            tokenizer_class=LlamaTokenizer,
            processor_class=AutoProcessor,
        )
    assert isinstance(model, models.TransformersVision)
    assert isinstance(model.model, LlavaForConditionalGeneration)
    assert isinstance(model.tokenizer.tokenizer, LlamaTokenizer)
    assert isinstance(model.processor, LlavaProcessor)

    # Test with model, tokenizer, processor kwargs, and device
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers_vision` function is deprecated",
    ):
        model = models.transformers_vision(
            model_name="trl-internal-testing/tiny-LlavaForConditionalGeneration",
            model_class=LlavaForConditionalGeneration,
            model_kwargs={"output_attentions": True},
            processor_kwargs={"num_additional_image_tokens": 1},
            device="cpu",
        )
    assert isinstance(model, models.TransformersVision)
    assert isinstance(model.model, LlavaForConditionalGeneration)
    assert isinstance(model.tokenizer.tokenizer, LlamaTokenizerFast)
    assert isinstance(model.processor, LlavaProcessor)
    assert model.model.config.output_attentions
    assert model.processor.num_additional_image_tokens == 1
    assert model.model.device.type == "cpu"


def test_transformers_vision_legacy_call_generation():
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers_vision` function is deprecated",
    ):
        model = models.transformers_vision(
            model_name="trl-internal-testing/tiny-LlavaForConditionalGeneration",
            model_class=LlavaForConditionalGeneration,
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
        "Hello, world!<image>",
        img_from_url(
            "https://upload.wikimedia.org/wikipedia/commons"
            + "/2/25/Siam_lilacpoint.jpg"
        ),
        10,
        "foo",
        2,
        length_penalty=0.5,
    )
    assert isinstance(result, str)
