from io import BytesIO
from urllib.request import urlopen

import pytest
from PIL import Image
from transformers import LlavaForConditionalGeneration

import outlines
from outlines.models.transformers_vision import transformers_vision


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    return transformers_vision(
        "trl-internal-testing/tiny-random-LlavaForConditionalGeneration",
        model_class=LlavaForConditionalGeneration,
        device="cpu",
    )


@pytest.fixture(scope="session")
def image(tmp_path_factory):
    def img_from_url(url):
        img_byte_stream = BytesIO(urlopen(url).read())
        return Image.open(img_byte_stream).convert("RGB")

    return img_from_url(
        "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
    )


def test_text_gen(model, image):
    assert model.model.device.type == "cpu"
    description_generator = outlines.generate.text(model)
    sequence = description_generator(
        "<|im_start|>user\n<image>\nWhat is this?<|im_end|>\n<|im_start|>assistant\n",
        [image],
        seed=10000,
        max_tokens=10,
    )
    assert isinstance(sequence, str)
