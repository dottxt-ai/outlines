import pytest
import io

import llama_cpp
from PIL import Image as PILImage
from llama_cpp import LogitsProcessorList

from outlines.inputs import Chat, Image
from outlines.models.llamacpp import LlamaCppTypeAdapter, LlamaCppTokenizer
from outlines.processors.structured import RegexLogitsProcessor


@pytest.fixture
def adapter():
    return LlamaCppTypeAdapter()


@pytest.fixture
def logits_processor():
    model = llama_cpp.Llama.from_pretrained(
        "M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )
    return RegexLogitsProcessor(
        regex_string=r"[a-z]+",
        tokenizer=LlamaCppTokenizer(model),
        tensor_library_name="numpy",
    )


@pytest.fixture
def image():
    width, height = 1, 1
    white_background = (255, 255, 255)
    image = PILImage.new("RGB", (width, height), white_background)

    # Save to an in-memory bytes buffer and read as png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)

    return image


def test_llamacpp_type_adapter_format_input(adapter, image):
    # Anything else than a string/Chat
    with pytest.raises(NotImplementedError):
        adapter.format_input(["Hello, world!"])

    # string
    assert adapter.format_input("Hello, world!") == "Hello, world!"

    # Chat
    messages = [
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hello, world!"},
    ]
    assert adapter.format_input(Chat(messages=messages)) == messages

    # Multi-modal (invalid)
    with pytest.raises(
        ValueError,
        match="LlamaCpp does not support multi-modal messages."
    ):
        adapter.format_input(Chat(messages=[
            {"role": "user", "content": ["prompt", Image(image)]},
        ]))


def test_llamacpp_type_adapter_format_output_type(adapter, logits_processor):
    formatted = adapter.format_output_type(logits_processor)
    assert isinstance(formatted, LogitsProcessorList)
    assert len(formatted) == 1
    assert isinstance(formatted[0], RegexLogitsProcessor)
