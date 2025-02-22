import re
from enum import Enum
from io import BytesIO
from urllib.request import urlopen

import pytest
from PIL import Image
from pydantic import BaseModel
from transformers import (
    Blip2ForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor,
)

import outlines
from outlines.models.transformers import TransformersVision
from outlines.types import Choice, JsonType, Regex

TEST_MODEL = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
TEST_CLIP_MODEL = "openai/clip-vit-base-patch32"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/71/2010-kodiak-bear-1.jpg",
]


@pytest.fixture
def images():
    def img_from_url(url):
        img_byte_stream = BytesIO(urlopen(url).read())
        return Image.open(img_byte_stream).convert("RGB")

    return [img_from_url(url) for url in IMAGE_URLS]


@pytest.fixture
def model():
    return outlines.from_transformers(
        LlavaForConditionalGeneration.from_pretrained(TEST_MODEL),
        AutoProcessor.from_pretrained(TEST_MODEL),
    )


def test_transformers_vision_instantiate_simple():
    model = outlines.from_transformers(
        Blip2ForConditionalGeneration.from_pretrained(TEST_MODEL),
        AutoProcessor.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, TransformersVision)


def test_transformers_vision_simple(model, images):
    result = model.generate(
        {"prompts": "<image>Describe this image in one sentence:", "images": images[0]},
        None,
    )
    assert isinstance(result, str)


def test_transformers_vision_call(model, images):
    result = model(
        {"prompts": "<image>Describe this image in one sentence:", "images": images[0]}
    )
    assert isinstance(result, str)


def test_transformers_vision_wrong_number_images(model, images):
    with pytest.raises(ValueError):
        a = model(
            {
                "prompts": "<image>Describe this image in one sentence:",
                "images": [images[0], images[1]],
            }
        )
        print(a)


def test_transformers_vision_wrong_input_type(model):
    with pytest.raises(NotImplementedError):
        model.generate("invalid input", None)


def test_transformers_inference_kwargs(model, images):
    result = model(
        {"prompts": "<image>Describe this image in one sentence:", "images": images[0]},
        max_new_tokens=100,
    )
    assert isinstance(result, str)


def test_transformers_invalid_inference_kwargs(model, images):
    with pytest.raises(ValueError):
        model(
            {
                "prompts": "<image>Describe this image in one sentence:",
                "images": images[0],
            },
            foo="bar",
        )


def test_transformers_several_images(model, images):
    result = model(
        {
            "prompts": "<image><image>Describe this image in one sentence:",
            "images": [images[0], images[1]],
        }
    )
    assert isinstance(result, str)


def test_transformers_vision_json(model, images):
    class Foo(BaseModel):
        name: str

    result = model(
        {"prompts": "<image>Give a name to this animal.", "images": images[0]},
        JsonType(Foo),
    )
    assert "name" in result


def test_transformers_vision_regex(model, images):
    result = model(
        {"prompts": "<image>How old is it?", "images": images[0]}, Regex(r"[0-9]")
    )

    assert isinstance(result, str)
    assert re.match(r"[0-9]", result)


def test_transformers_vision_choice(model, images):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model(
        {"prompts": "<image>Is it a cat or a dog?", "images": images[0]}, Choice(Foo)
    )

    assert isinstance(result, str)
    assert result in ["cat", "dog"]


def test_transformers_vision_batch_samples(model, images):
    result = model(
        {"prompts": "<image>Describe this image in one sentence.", "images": images[0]},
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        {
            "prompts": [
                "<image>Describe this image in one sentence.",
                "<image>Describe this image in one sentence.",
            ],
            "images": [images[0], images[1]],
        }
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        {
            "prompts": [
                "<image>Describe this image in one sentence.",
                "<image>Describe this image in one sentence.",
            ],
            "images": [images[0], images[1]],
        },
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2


def test_transformers_vision_batch_samples_constrained(model, images):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model(
        {"prompts": "<image>Describe this image in one sentence.", "images": images[0]},
        Choice(Foo),
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert item in ["cat", "dog"]
    result = model(
        {
            "prompts": [
                "<image>Describe this image in one sentence.",
                "<image>Describe this image in one sentence.",
            ],
            "images": [images[0], images[1]],
        },
        Choice(Foo),
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert item in ["cat", "dog"]
    result = model(
        {
            "prompts": [
                "<image>Describe this image in one sentence.",
                "<image>Describe this image in one sentence.",
            ],
            "images": [images[0], images[1]],
        },
        Choice(Foo),
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2
        assert item[0] in ["cat", "dog"]
        assert item[1] in ["cat", "dog"]
