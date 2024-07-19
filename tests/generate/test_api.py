from io import BytesIO
from urllib.request import urlopen

import pytest
from PIL import Image  # type: ignore

from outlines.generate.api import VisionSequenceGeneratorAdapter

IMG_URI = "https://upload.wikimedia.org/wikipedia/en/a/a9/Example.jpg"
PIL_IMG = Image.open(BytesIO(urlopen(IMG_URI).read())).convert("RGB")


@pytest.mark.parametrize(
    "prompts,media,type_error",
    [
        ("single prompt", [PIL_IMG], False),
        (["prompt0", "prompt1"], [[PIL_IMG], [PIL_IMG]], False),
        ("single prompt", [PIL_IMG, PIL_IMG], False),
        (["prompt0", "prompt1"], [[PIL_IMG, PIL_IMG], [PIL_IMG]], False),
        ("single prompt", "this isn't an image, it's a string", True),
        ("single prompt", PIL_IMG, True),
        (["prompt0", "prompt1"], [PIL_IMG], True),
        (["prompt0", "prompt1"], [[PIL_IMG]], True),
        (["prompt0", "prompt1"], [[[PIL_IMG]], [[PIL_IMG]]], True),
    ],
)
def test_vision_sequence_generator_validate_types(prompts, media, type_error):
    """Ensure inputs are validated correctly"""
    if type_error:
        with pytest.raises(TypeError):
            VisionSequenceGeneratorAdapter._validate_prompt_media_types(prompts, media)
    else:
        VisionSequenceGeneratorAdapter._validate_prompt_media_types(prompts, media)
