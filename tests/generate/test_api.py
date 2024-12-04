from io import BytesIO
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image  # type: ignore

from outlines.generate.api import (
    AudioSequenceGeneratorAdapter,
    VisionSequenceGeneratorAdapter,
)

IMG_URI = "https://upload.wikimedia.org/wikipedia/en/a/a9/Example.jpg"
PIL_IMG = Image.open(BytesIO(urlopen(IMG_URI).read())).convert("RGB")
AUDIO_ARRAY = np.array([1, 2])


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


@pytest.mark.parametrize(
    "prompts,media,type_error",
    [
        ("single prompt", [AUDIO_ARRAY], False),
        (["single prompt"], [AUDIO_ARRAY], False),
        (["prompt0", "prompt1"], [AUDIO_ARRAY, AUDIO_ARRAY], False),
        ("single prompt", [AUDIO_ARRAY, AUDIO_ARRAY], False),
        ("single prompt", "this isn't an audio, it's a string", True),
        ("single prompt", AUDIO_ARRAY, True),
        (["prompt0", "prompt1"], [AUDIO_ARRAY], False),
        ("prompt0", [[AUDIO_ARRAY]], True),
        (["prompt0", "prompt1"], [[AUDIO_ARRAY]], True),
        (["prompt0", "prompt1"], [[[AUDIO_ARRAY]], [[AUDIO_ARRAY]]], True),
    ],
)
def test_audio_sequence_generator_validate_types(prompts, media, type_error):
    """Ensure inputs are validated correctly"""
    if type_error:
        with pytest.raises(TypeError):
            AudioSequenceGeneratorAdapter._validate_prompt_media_types(prompts, media)
    else:
        AudioSequenceGeneratorAdapter._validate_prompt_media_types(prompts, media)
