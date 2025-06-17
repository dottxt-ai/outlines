# we only test vision models here as audio models are too heavy to run on CI

import re
from enum import Enum
from io import BytesIO
from urllib.request import urlopen

import pytest
from PIL import Image
from pydantic import BaseModel
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

import outlines
from outlines.models.transformers import (
    TransformersMultiModal,
    TransformerTokenizer,
    TransformersMultiModalTypeAdapter,
)
from outlines.types import Regex

TEST_MODEL = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
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
        LlavaForConditionalGeneration.from_pretrained(TEST_MODEL),
        AutoProcessor.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, TransformersMultiModal)
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert isinstance(model.type_adapter, TransformersMultiModalTypeAdapter)
    assert model.tensor_library_name == "torch"


def test_transformers_vision_simple(model, images):
    result = model.generate(
        {"text": "<image>Describe this image in one sentence:", "images": images[0]},
        None,
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_vision_call(model, images):
    result = model(
        {"text": "<image>Describe this image in one sentence:", "images": images[0]},
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_vision_wrong_number_images(model, images):
    with pytest.raises(ValueError):
        model(
            {
                "text": "<image>Describe this image in one sentence:",
                "images": [images[0], images[1]],
            }
        )


def test_transformers_vision_wrong_input_type(model):
    with pytest.raises(NotImplementedError):
        model.generate("invalid input", None)


def test_transformers_inference_kwargs(model, images):
    result = model(
        {"text": "<image>Describe this image in one sentence:", "images": images[0]},
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_invalid_inference_kwargs(model, images):
    with pytest.raises(ValueError):
        model(
            {
                "text": "<image>Describe this image in one sentence:",
                "images": images[0],
            },
            foo="bar",
        )


def test_transformers_several_images(model, images):
    result = model(
        {
            "text": "<image><image>Describe this image in one sentence:",
            "images": [images[0], images[1]],
        },
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_vision_json(model, images):
    class Foo(BaseModel):
        name: str

    result = model(
        {"text": "<image>Give a name to this animal.", "images": images[0]},
        Foo,
        max_new_tokens=10,
    )
    assert "name" in result


def test_transformers_vision_regex(model, images):
    result = model(
        {"text": "<image>How old is it?", "images": images[0]},
        Regex(r"[0-9]")
    )

    assert isinstance(result, str)
    assert re.match(r"[0-9]", result)


def test_transformers_vision_choice(model, images):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model(
        {"text": "<image>Is it a cat or a dog?", "images": images[0]},
        Foo,
    )

    assert isinstance(result, str)
    assert result in ["cat", "dog"]


def test_transformers_vision_batch_samples(model, images):
    result = model(
        {"text": "<image>Describe this image in one sentence.", "images": images[0]},
        num_return_sequences=2,
        num_beams=2,
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        {
            "text": [
                "<image>Describe this image in one sentence.",
                "<image>Describe this image in one sentence.",
            ],
            "images": [images[0], images[1]],
        },
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        {
            "text": [
                "<image>Describe this image in one sentence.",
                "<image>Describe this image in one sentence.",
            ],
            "images": [images[0], images[1]],
        },
        num_return_sequences=2,
        num_beams=2,
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2
