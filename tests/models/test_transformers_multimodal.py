# we only test vision models here as audio models are too heavy to run on CI

import io
import re
from enum import Enum

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


@pytest.fixture
def image():
    width, height = 256, 256
    blue_background = (0, 0, 255)
    image = PILImage.new("RGB", (width, height), blue_background)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)

    return image


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


def test_transformers_multimodal_simple(model, image):
    result = model.generate(
        ["<image>Describe this image in one sentence:", Image(image)],
        None,
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_multimodal_call(model, image):
    result = model(
        ["<image>Describe this image in one sentence:", Image(image)],
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_multimodal_wrong_number_image(model, image):
    with pytest.raises(ValueError):
        model(
            [
                "<image>Describe this image in one sentence:",
                Image(image),
                Image(image),
            ],
        )


def test_transformers_multimodal_wrong_input_type(model):
    with pytest.raises(TypeError):
        model.generate("invalid input", None)


def test_transformers_multimodal_chat(model, image):
    result = model(
        Chat(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    "Describe this image in one sentence:<image>",
                    Image(image),
                ],
            },
        ]),
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_inference_kwargs(model, image):
    result = model(
        ["<image>Describe this image in one sentence:", Image(image)],
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_invalid_inference_kwargs(model, image):
    with pytest.raises(ValueError):
        model(
            [
                "<image>Describe this image in one sentence:",
                Image(image),
            ],
            foo="bar",
        )


def test_transformers_several_image(model, image):
    result = model(
        [
            "<image><image>Describe this image in one sentence:",
            Image(image),
            Image(image),
        ],
        max_new_tokens=2,
    )
    assert isinstance(result, str)


def test_transformers_multimodal_json(model, image):
    class Foo(BaseModel):
        name: str

    result = model(
        ["<image>Give the name of the color.", Image(image)],
        Foo,
        max_new_tokens=10,
    )
    assert "name" in result


def test_transformers_multimodal_regex(model, image):
    result = model(
        ["<image>How warn is the color from 0 to 9?", Image(image)],
        Regex(r"[0-9]")
    )

    assert isinstance(result, str)
    assert re.match(r"[0-9]", result)


def test_transformers_multimodal_choice(model, image):
    class Foo(Enum):
        white = "white"
        blue = "blue"

    result = model(
        ["<image>Is it a white or a blue?", Image(image)],
        Foo,
    )

    assert isinstance(result, str)
    assert result in ["white", "blue"]


def test_transformers_multimodal_multiple_samples(model, image):
    result = model(
        ["<image>Describe this image in one sentence.", Image(image)],
        num_return_sequences=2,
        num_beams=2,
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2


def test_transformers_multimodal_batch(model, image):
    result = model.batch(
        [
            ["<image>Describe this image in one sentence.", Image(image)],
            ["<image>Describe this image in one sentence.", Image(image)],
        ],
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2

    result = model.batch(
        [
            ["<image>Describe this image in one sentence.<image>", Image(image), Image(image)],
            ["<image>Describe this image in one sentence.<image>", Image(image), Image(image)],
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
                        "Describe this image in one sentence:<image>",
                        Image(image),
                    ],
                },
            ]),
            Chat(messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        "Describe this image in one sentence:<image>",
                        Image(image),
                    ],
                },
            ]),
        ],
        max_new_tokens=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2


def test_transformers_multimodal_deprecated_input_type(model, image):
    with pytest.warns(DeprecationWarning):
        result = model.generate(
            {
                "text": "<image>Describe this image in one sentence:",
                "image": image,
            },
            None,
            max_new_tokens=2,
        )
        assert isinstance(result, str)
