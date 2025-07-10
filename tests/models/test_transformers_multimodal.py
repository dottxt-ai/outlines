# we only test vision models here as audio models are too heavy to run on CI

import re
from enum import Enum
from io import BytesIO
from urllib.request import urlopen

import pytest
from PIL import Image as PILImage
from pydantic import BaseModel
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

import outlines
from outlines.inputs import Chat, Image
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
        image = PILImage.open(img_byte_stream).convert("RGB")
        image.format = "PNG"
        return image

    return [img_from_url(url) for url in IMAGE_URLS]


@pytest.fixture
def model():
    model = outlines.from_transformers(
        LlavaForConditionalGeneration.from_pretrained(TEST_MODEL),
        AutoProcessor.from_pretrained(TEST_MODEL),
    )
    chat_template = '{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}'
    model.type_adapter.tokenizer.chat_template = chat_template

    return model


def test_transformers_multimodal_instantiate_simple():
    model = outlines.from_transformers(
        LlavaForConditionalGeneration.from_pretrained(TEST_MODEL),
        AutoProcessor.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, TransformersMultiModal)
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert isinstance(model.type_adapter, TransformersMultiModalTypeAdapter)
    assert model.tensor_library_name == "torch"


def test_transformers_multimodal_simple(model, images):
    result = model.generate(
        ["<image>Describe this image in one sentence:", Image(images[0])],
        None,
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_multimodal_call(model, images):
    result = model(
        ["<image>Describe this image in one sentence:", Image(images[0])],
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_multimodal_wrong_number_images(model, images):
    with pytest.raises(ValueError):
        model(
            [
                "<image>Describe this image in one sentence:",
                Image(images[0]),
                Image(images[1]),
            ],
        )


def test_transformers_multimodal_wrong_input_type(model):
    with pytest.raises(TypeError):
        model.generate("invalid input", None)


def test_transformers_multimodal_chat(model, images):
    result = model(
        Chat(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    "What's on this image?<image>",
                    Image(images[0]),
                ],
            },
        ]),
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_inference_kwargs(model, images):
    result = model(
        ["<image>Describe this image in one sentence:", Image(images[0])],
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_invalid_inference_kwargs(model, images):
    with pytest.raises(ValueError):
        model(
            [
                "<image>Describe this image in one sentence:",
                Image(images[0]),
            ],
            foo="bar",
        )


def test_transformers_several_images(model, images):
    result = model(
        [
            "<image><image>Describe this image in one sentence:",
            Image(images[0]),
            Image(images[1]),
        ],
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_multimodal_json(model, images):
    class Foo(BaseModel):
        name: str

    result = model(
        ["<image>Give a name to this animal.", Image(images[0])],
        Foo,
        max_new_tokens=10,
    )
    assert "name" in result


def test_transformers_multimodal_regex(model, images):
    result = model(
        ["<image>How old is it?", Image(images[0])],
        Regex(r"[0-9]")
    )

    assert isinstance(result, str)
    assert re.match(r"[0-9]", result)


def test_transformers_multimodal_choice(model, images):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model(
        ["<image>Is it a cat or a dog?", Image(images[0])],
        Foo,
    )

    assert isinstance(result, str)
    assert result in ["cat", "dog"]


def test_transformers_multimodal_multiple_samples(model, images):
    result = model(
        ["<image>Describe this image in one sentence.", Image(images[0])],
        num_return_sequences=2,
        num_beams=2,
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2


def test_transformers_multimodal_batch(model, images):
    result = model.batch(
        [
            ["<image>Describe this image in one sentence.", Image(images[0])],
            ["<image>Describe this image in one sentence.", Image(images[0])],
        ],
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2

    result = model.batch(
        [
            ["<image>Describe this image in one sentence.<image>", Image(images[0]), Image(images[1])],
            ["<image>Describe this image in one sentence.<image>", Image(images[0]), Image(images[1])],
        ],
        num_return_sequences=2,
        num_beams=2,
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2

    result = model.batch(
        [
            Chat(messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        "What's on this image?<image>",
                        Image(images[0]),
                    ],
                },
            ]),
            Chat(messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        "What's on this image?<image>",
                        Image(images[1]),
                    ],
                },
            ]),
        ],
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2


def test_transformers_multimodal_deprecated_input_type(model, images):
    with pytest.warns(DeprecationWarning):
        result = model.generate(
            {
                "text": "<image>Describe this image in one sentence:",
                "images": images[0],
            },
            None,
            max_new_tokens=2,
        )
        assert isinstance(result, str)
