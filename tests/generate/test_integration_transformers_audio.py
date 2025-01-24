from io import BytesIO
from urllib.request import urlopen

import librosa
import pytest
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2AudioForConditionalGeneration,
    Qwen2TokenizerFast,
    ViTModel,
)

import outlines
from outlines.models.transformers_audio import transformers_audio

AUDIO_URLS = [
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav",
]
QWEN2_AUDIO_SAMPLING_RATE = 16000

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def audio_from_url(url):
    audio_byte_stream = BytesIO(urlopen(url).read())
    return librosa.load(audio_byte_stream, sr=QWEN2_AUDIO_SAMPLING_RATE)[0]


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    return transformers_audio(
        "yujiepan/qwen2-audio-tiny-random",
        model_class=Qwen2AudioForConditionalGeneration,
        device="cpu",
    )


@pytest.fixture(scope="session")
def processor(tmp_path_factory):
    return AutoProcessor.from_pretrained("yujiepan/qwen2-audio-tiny-random")


def test_single_audio_text_gen(model, processor):
    conversation = [
        {
            "role": "user",
            "content": [
                {"audio"},
                {"type": "text", "text": "What's that sound?"},
            ],
        },
    ]
    generator = outlines.generate.text(model)
    sequence = generator(
        processor.apply_chat_template(conversation),
        [audio_from_url(AUDIO_URLS[0])],
        seed=10000,
        max_tokens=10,
    )
    assert isinstance(sequence, str)


def test_multi_audio_text_gen(model, processor):
    """If the length of audio tags and number of audios we pass are > 1 and equal,
    we should yield a successful generation.
    """
    conversation = [
        {
            "role": "user",
            "content": [{"audio"} for _ in range(len(AUDIO_URLS))]
            + [
                {
                    "type": "text",
                    "text": "Did a human make one of the audio recordings?",
                }
            ],
        },
    ]
    generator = outlines.generate.text(model)
    sequence = generator(
        processor.apply_chat_template(conversation),
        [audio_from_url(url) for url in AUDIO_URLS],
        seed=10000,
        max_tokens=10,
    )
    assert isinstance(sequence, str)


def test_mismatched_audio_text_gen(model, processor):
    """If the length of audio tags and number of audios we pass are unequal,
    we should raise an error.
    """
    generator = outlines.generate.text(model)

    conversation = [
        {
            "role": "user",
            "content": [
                {"audio"},
                {"type": "text", "text": "I'm passing 2 audios, but only 1 audio tag"},
            ],
        },
    ]
    with pytest.raises(ValueError):
        _ = generator(
            processor.apply_chat_template(conversation),
            [audio_from_url(i) for i in AUDIO_URLS],
            seed=10000,
            max_tokens=10,
        )

    conversation = [
        {
            "role": "user",
            "content": [
                {"audio"},
                {"audio"},
                {"type": "text", "text": "I'm passing 2 audio tags, but only 1 audio"},
            ],
        },
    ]
    with pytest.raises(ValueError):
        _ = generator(
            processor.apply_chat_template(conversation),
            [audio_from_url(AUDIO_URLS[0])],
            seed=10000,
            max_tokens=10,
        )


def test_single_audio_choice(model, processor):
    conversation = [
        {
            "role": "user",
            "content": [
                {"audio"},
                {"type": "text", "text": "What's that sound?"},
            ],
        },
    ]
    choices = ["dog barking", "glass breaking"]
    generator = outlines.generate.choice(model, choices)
    sequence = generator(
        processor.apply_chat_template(conversation),
        [audio_from_url(AUDIO_URLS[0])],
        seed=10000,
        max_tokens=10,
    )
    assert isinstance(sequence, str)
    assert sequence in choices


def test_tokenizer_class():
    model = transformers_audio(
        "yujiepan/qwen2-audio-tiny-random",
        model_class=Qwen2AudioForConditionalGeneration,
        tokenizer_class=AutoTokenizer,
        device="cpu",
    )
    assert isinstance(model.processor.tokenizer, Qwen2TokenizerFast)


def test_no_tokenizer_in_processor():
    with pytest.raises(KeyError):
        transformers_audio(
            "google/vit-base-patch16-224",  # only way to have a processor with no tokenizer is loading a non-text model
            model_class=ViTModel,
            device="cpu",
        )
