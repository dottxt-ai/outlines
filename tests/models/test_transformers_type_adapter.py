import io
import pytest
from unittest.mock import MagicMock

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
    tokenizer = MagicMock()
    chat_template = "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    tokenizer.chat_template = chat_template

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        res = ""
        for msg in messages:
            res += f"{msg['role']}: {msg['content']}"
        if add_generation_prompt:
            res += "assistant:"
        return res

    tokenizer.apply_chat_template.side_effect = apply_chat_template

    type_adapter = TransformersTypeAdapter(tokenizer=tokenizer, has_chat_template=True)
    return type_adapter

@pytest.fixture
def logits_processor():
    index = MagicMock()
    index.tensor_library_name = "torch"
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

    # string with chat template
    # The fixture sets a chat template, so it should be formatted
    assert adapter.format_input("Hello, world!") == "user: Hello, world!assistant:"

    # string without chat template
    adapter.has_chat_template = False
    assert adapter.format_input("Hello, world!") == "Hello, world!"

    # chat
    # Restore chat template for chat test
    adapter.has_chat_template = True
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
