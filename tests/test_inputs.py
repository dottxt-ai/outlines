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
