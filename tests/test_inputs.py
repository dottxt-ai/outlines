"""Unit tests for the inputs module."""

import base64
import tempfile
from io import BytesIO
from typing import Dict, List, Any

import pytest
from PIL import Image as PILImage

from outlines.inputs import Image, Video, Audio, Chat


@pytest.fixture
def image_input():
    image = PILImage.new("RGB", (100, 100), color="red")
    image.format = "PNG"
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return Image(image=image)


def test_image_initialization():
    # png
    image = PILImage.new("RGB", (100, 100), color="red")
    image.format = "PNG"
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_input = Image(image=image)

    assert image_input.image == image
    assert image_input.image_format == "image/png"
    assert image_input.image_str == base64.b64encode(buffer.getvalue()).decode("utf-8")

    # jpeg
    image = PILImage.new("RGB", (100, 100), color="blue")
    image.format = "JPEG"
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_input = Image(image=image)

    assert image_input.image == image
    assert image_input.image_format == "image/jpeg"
    assert image_input.image_str == base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_image_initialization_invalid():
    """Test that Image initialization fails when image has no format."""
    # No format
    image = PILImage.new("RGB", (100, 100), color="yellow")
    with pytest.raises(TypeError, match="Could not read the format of the image"):
        Image(image=image)

    # Empty string format
    image = PILImage.new("RGB", (100, 100), color="orange")
    image.format = ""
    with pytest.raises(TypeError, match="Could not read the format of the image"):
        Image(image=image)


def test_image_initialization_bytes():
    image = PILImage.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()

    image_input = Image(image=raw)

    assert image_input.is_bytes()
    assert not image_input.is_pil()
    assert not image_input.is_url()
    assert image_input.image_format == "image/png"
    assert image_input.image_str == base64.b64encode(raw).decode("utf-8")

    pil_image = image_input.get_pil_image()
    assert pil_image.size == (100, 100)


def test_image_initialization_bytes_invalid():
    with pytest.raises(TypeError, match="Could not read the image bytes"):
        Image(image=b"not-an-image")


def test_image_initialization_url():
    url = "https://example.com/image.png"
    image_input = Image(image=url)

    assert image_input.is_url()
    assert not image_input.is_pil()
    assert not image_input.is_bytes()
    assert image_input.image == url
    assert image_input.image_str is None
    assert image_input.image_format is None

    with pytest.raises(ValueError, match="URL images are not supported"):
        image_input.get_pil_image()


def test_image_initialization_invalid_type():
    with pytest.raises(TypeError, match="Unsupported image type"):
        Image(image=123)  # type: ignore

    with pytest.raises(TypeError, match="Unsupported image type"):
        Image(image="/tmp/local-path.png")  # type: ignore


def test_image_is_helpers_pil(image_input):
    assert image_input.is_pil()
    assert not image_input.is_bytes()
    assert not image_input.is_url()
    assert image_input.get_pil_image() == image_input.image


def test_image_url_passthrough_adapters():
    from outlines.models.anthropic import AnthropicTypeAdapter
    from outlines.models.gemini import GeminiTypeAdapter
    from outlines.models.mistral import MistralTypeAdapter
    from outlines.models.ollama import OllamaTypeAdapter
    from outlines.models.openai import OpenAITypeAdapter

    url = "https://example.com/image.png"
    url_image = Image(image=url)

    openai_part = OpenAITypeAdapter()._create_img_content(url_image)
    assert openai_part["image_url"]["url"] == url

    anthropic_part = AnthropicTypeAdapter()._create_img_content(url_image)
    assert anthropic_part["source"]["type"] == "url"
    assert anthropic_part["source"]["url"] == url

    mistral_messages = MistralTypeAdapter()._create_message_content(["prompt", url_image])
    assert mistral_messages[1]["image_url"]["url"] == url

    with pytest.raises(ValueError, match="URL images are not supported"):
        GeminiTypeAdapter()._create_img_part(url_image)

    with pytest.raises(ValueError, match="URL images are not supported"):
        OllamaTypeAdapter().format_input(["prompt", url_image])


def test_image_bytes_adapters():
    from outlines.models.anthropic import AnthropicTypeAdapter
    from outlines.models.gemini import GeminiTypeAdapter
    from outlines.models.ollama import OllamaTypeAdapter
    from outlines.models.openai import OpenAITypeAdapter

    pil_image = PILImage.new("RGB", (10, 10), color="red")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    bytes_image = Image(image=raw)

    openai_part = OpenAITypeAdapter()._create_img_content(bytes_image)
    assert openai_part["image_url"]["url"].startswith("data:image/png;base64,")

    anthropic_part = AnthropicTypeAdapter()._create_img_content(bytes_image)
    assert anthropic_part["source"]["type"] == "base64"

    gemini_part = GeminiTypeAdapter()._create_img_part(bytes_image)
    assert gemini_part["inline_data"]["mime_type"] == "image/png"

    ollama_messages = OllamaTypeAdapter().format_input(["prompt", bytes_image])
    assert ollama_messages[0]["images"] == [bytes_image.image_str]


def test_image_url_transformers_error():
    from outlines.models.transformers import TransformersMultiModalTypeAdapter

    adapter = TransformersMultiModalTypeAdapter(tokenizer=None)
    url_image = Image(image="https://example.com/image.png")
    with pytest.raises(ValueError, match="URL images are not supported"):
        adapter.format_input(["prompt", url_image])


def test_image_bytes_transformers():
    from outlines.models.transformers import TransformersMultiModalTypeAdapter

    pil_image = PILImage.new("RGB", (10, 10), color="red")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    bytes_image = Image(image=buffer.getvalue())

    adapter = TransformersMultiModalTypeAdapter(tokenizer=None)
    result = adapter.format_input(["prompt", bytes_image])
    assert result["text"] == "prompt"
    assert len(result["images"]) == 1
    assert isinstance(result["images"][0], PILImage.Image)
    assert result["images"][0].size == (10, 10)
    assert result["images"][0].getpixel((0, 0)) == (255, 0, 0)


def test_video_initialization():
    video = "foo"
    video_input = Video(video=video)
    assert video_input.video == video


def test_audio_initialization():
    audio = "foo"
    audio_input = Audio(audio=audio)
    assert audio_input.audio == audio


def test_chat_initialization():
    # Empty
    chat = Chat()
    assert chat.messages == []
    assert len(chat.messages) == 0
    assert str(chat) == ""
    assert repr(chat) == "Chat(messages=[])"

    # With messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    chat = Chat(messages=messages)
    assert chat.messages == messages
    assert len(chat.messages) == 3
    assert str(chat) == "{'role': 'system', 'content': 'You are a helpful assistant.'}\n{'role': 'user', 'content': 'Hello!'}\n{'role': 'assistant', 'content': 'Hi there!'}"
    assert repr(chat) == "Chat(messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello!'}, {'role': 'assistant', 'content': 'Hi there!'}])"


def test_chat_append():
    chat = Chat(messages=[])
    message = {"role": "user", "content": "Hello"}
    chat.append(message)
    assert len(chat.messages) == 1
    assert chat.messages[0] == message

def test_chat_extend():
    chat = Chat(messages=[])
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    chat.extend(messages)
    assert len(chat.messages) == 2
    assert chat.messages == messages

def test_chat_pop():
    # Pop from non-empty chat
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    chat = Chat(messages=messages.copy())
    popped_message = chat.pop()
    assert popped_message == {"role": "assistant", "content": "Hi"}
    assert len(chat.messages) == 1
    assert chat.messages[0] == {"role": "user", "content": "Hello"}

    # Pop from empty chat
    chat = Chat(messages=[])
    with pytest.raises(IndexError):
        chat.pop()


def test_chat_add_system_message(image_input):
    # Add a string
    chat = Chat(messages=[])
    chat.add_system_message("You are a helpful assistant.")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "system"
    assert chat.messages[0]["content"] == "You are a helpful assistant."

    # Add a list
    chat = Chat(messages=[])
    chat.add_system_message(["prompt", image_input])
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "system"
    assert chat.messages[0]["content"] == ["prompt", image_input]

    # Add a list of dict items with explicit types
    chat = Chat(messages=[])
    chat.add_system_message([{"type": "text", "text": "prompt"}, {"type": "image", "image": image_input}])
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "system"
    assert chat.messages[0]["content"] == [{"type": "text", "text": "prompt"}, {"type": "image", "image": image_input}]


def test_add_user_message_string(image_input):
    # Add a string
    chat = Chat(messages=[])
    chat.add_user_message("Hello, how are you?")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "user"
    assert chat.messages[0]["content"] == "Hello, how are you?"

    # Add a list
    chat = Chat(messages=[])
    chat.add_user_message(["prompt", image_input])
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "user"
    assert chat.messages[0]["content"] == ["prompt", image_input]

    # Add a list of dict items with explicit types
    chat = Chat(messages=[])
    chat.add_user_message([{"type": "text", "text": "prompt"}, {"type": "image", "image": image_input}])
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "user"
    assert chat.messages[0]["content"] == [{"type": "text", "text": "prompt"}, {"type": "image", "image": image_input}]


def test_add_assistant_message_string(image_input):
    # Add a string
    chat = Chat(messages=[])
    chat.add_assistant_message("I'm doing well, thank you!")
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "assistant"
    assert chat.messages[0]["content"] == "I'm doing well, thank you!"

    # Add a list
    chat = Chat(messages=[])
    chat.add_assistant_message(["prompt", image_input])
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "assistant"
    assert chat.messages[0]["content"] == ["prompt", image_input]

    # Add a list of dict items with explicit types
    chat = Chat(messages=[])
    chat.add_assistant_message([{"type": "text", "text": "prompt"}, {"type": "image", "image": image_input}])
    assert len(chat.messages) == 1
    assert chat.messages[0]["role"] == "assistant"
    assert chat.messages[0]["content"] == [{"type": "text", "text": "prompt"}, {"type": "image", "image": image_input}]
