import pytest
import io

from llama_cpp import LogitsProcessorList
from PIL import Image as PILImage
from outlines_core import Index, Vocabulary

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor
from outlines.inputs import Chat, Image
from outlines.models.llamacpp import LlamaCppTypeAdapter


@pytest.fixture
def adapter():
    return LlamaCppTypeAdapter()


@pytest.fixture
def logits_processor():
    vocabulary = Vocabulary.from_pretrained("openai-community/gpt2")
    index = Index(r"[0-9]{3}", vocabulary)
    return OutlinesCoreLogitsProcessor(index, "numpy")


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
    assert formatted[0].index == logits_processor.index
    assert formatted[0].tensor_library_name == logits_processor.tensor_library_name
