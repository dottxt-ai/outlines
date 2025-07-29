import io
import pytest

import transformers
from transformers import LogitsProcessorList
from outlines_core import Index, Vocabulary
from PIL import Image as PILImage

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor
from outlines.inputs import Chat, Image
from outlines.models.transformers import TransformersTypeAdapter


MODEL_NAME = "erwanf/gpt2-mini"


@pytest.fixture
def adapter():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    type_adapter = TransformersTypeAdapter(tokenizer=tokenizer)
    chat_template = '{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}'
    type_adapter.tokenizer.chat_template = chat_template

    return type_adapter

@pytest.fixture
def logits_processor():
    vocabulary = Vocabulary.from_pretrained("openai-community/gpt2")
    index = Index(r"[0-9]{3}", vocabulary)
    return OutlinesCoreLogitsProcessor(index, "torch")

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



def test_transformers_type_adapter_format_input(adapter, image):
    # invalid input
    with pytest.raises(TypeError, match="is not available."):
        adapter.format_input(["prompt", Image(image)])

    # string
    assert adapter.format_input("Hello, world!") == "Hello, world!"

    # chat
    assert isinstance(adapter.format_input(Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hello, world!"},
    ])), str)


def test_transformers_type_adapter_format_output_type(
    adapter, logits_processor
):
    formatted = adapter.format_output_type(logits_processor)
    assert isinstance(formatted, LogitsProcessorList)
    assert formatted[0].index == logits_processor.index
    assert formatted[0].tensor_library_name == logits_processor.tensor_library_name

    formatted = adapter.format_output_type(None)
    assert formatted is None
