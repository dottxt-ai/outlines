import pytest

from PIL import Image as PILImage
from outlines_core import Index, Vocabulary
from transformers import (
    AutoProcessor,
    LogitsProcessorList,
)

from outlines.inputs import Audio, Chat, Image, Video
from outlines.models.transformers import TransformersMultiModalTypeAdapter
from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor


MODEL_NAME = "trl-internal-testing/tiny-LlavaForConditionalGeneration"


@pytest.fixture
def adapter():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer
    type_adapter = TransformersMultiModalTypeAdapter(tokenizer=tokenizer)

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
    image.format = "PNG"

    return image


@pytest.fixture
def video():
    # Simple mock video data
    return "mock_video_data"


@pytest.fixture
def audio():
    # Simple mock audio data
    return "mock_audio_data"


def test_transformers_multimodal_type_adapter_format_input(adapter, image):
    with pytest.raises(TypeError):
        adapter.format_input("hello")

    with pytest.raises(TypeError):
        adapter.format_input({"foo": "bar"})

    with pytest.raises(ValueError, match="All assets must be of the same type"):
        adapter.format_input(["foo", Image(image), Video("")])

    class MockAsset:
        pass

    with pytest.raises(ValueError, match="Unsupported asset type"):
        adapter.format_input(["foo", MockAsset()])

    image_asset = Image(image)
    assert adapter.format_input(["foo", image_asset]) == {
        "text": "foo",
        "images": [image_asset.image],
    }

    chat_prompt = Chat(messages=[
        {"role": "system", "content": "foo"},
        {"role": "user", "content": ["bar", image_asset]},
    ])
    result = adapter.format_input(chat_prompt)
    assert isinstance(result, dict)
    assert isinstance(result["text"], str)
    assert isinstance(result["images"], list)
    assert len(result["images"]) == 1
    assert result["images"][0] == image_asset.image

    chat_prompt = Chat(messages=[
        {"role": "system", "content": "foo"},
        {"role": "user", "content": [{"type": "text", "text": "bar"}, {"type": "image", "image": image_asset}]},
    ])
    result = adapter.format_input(chat_prompt)
    assert isinstance(result, dict)
    assert isinstance(result["text"], str)
    assert isinstance(result["images"], list)
    assert len(result["images"]) == 1
    assert result["images"][0] == image_asset.image




def test_transformers_multimodal_type_adapter_format_input_empty_assets(adapter):
    result = adapter.format_input(["Just text prompt"])
    assert result == {"text": "Just text prompt"}


def test_transformers_multimodal_type_adapter_format_input_chat_invalid_asset_type(adapter, image):
    class MockAsset:
        pass

    chat_prompt = Chat(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image", "image": MockAsset()}  # Wrong type
        ]}
    ])

    with pytest.raises(ValueError, match="Assets must be of type"):
        adapter.format_input(chat_prompt)


def test_transformers_multimodal_type_adapter_format_input_chat_unsupported_content_type(adapter):
    chat_prompt = Chat(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "unsupported", "data": "some_data"}  # Unsupported type
        ]}
    ])

    with pytest.raises(ValueError, match="Content must be 'text'"):
        adapter.format_input(chat_prompt)


def test_transformers_multimodal_type_adapter_format_output_type(
    adapter, logits_processor
):
    formatted = adapter.format_output_type(logits_processor)
    assert isinstance(formatted, LogitsProcessorList)
    assert len(formatted) == 1
    assert formatted[0].index == logits_processor.index
    assert formatted[0].tensor_library_name == logits_processor.tensor_library_name

    formatted = adapter.format_output_type(None)
    assert formatted is None


def test_transformers_multimodal_type_adapter_format_input_chat_missing_asset_key(adapter, image):
    image_asset = Image(image)

    # Test missing 'image' key when type is 'image'
    chat_prompt = Chat(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "txt": image_asset}  # Wrong key: 'txt' instead of 'image'
        ]}
    ])

    with pytest.raises(ValueError, match="Item with type 'image' must contain a 'image' key"):
        adapter.format_input(chat_prompt)

    # Test missing 'video' key when type is 'video'
    video_asset = Video("dummy_video")
    chat_prompt = Chat(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this video?"},
            {"type": "video", "vid": video_asset}  # Wrong key: 'vid' instead of 'video'
        ]}
    ])

    with pytest.raises(ValueError, match="Item with type 'video' must contain a 'video' key"):
        adapter.format_input(chat_prompt)


def test_transformers_multimodal_type_adapter_format_input_chat_missing_type_key(adapter, image):
    image_asset = Image(image)

    chat_prompt = Chat(messages=[
        {"role": "user", "content": [
            {"text": "What's in this image?"},  # Missing 'type' key
            {"type": "image", "image": image_asset}
        ]}
    ])

    with pytest.raises(ValueError, match="Each item in the content list must be a dictionary with a 'type' key"):
        adapter.format_input(chat_prompt)


def test_transformers_multimodal_type_adapter_format_input_invalid_content_type(adapter):
    chat_prompt = Chat(messages=[
        {"role": "user", "content": 42}  # Invalid content type (integer)
    ])

    with pytest.raises(ValueError, match="Invalid content type"):
        adapter.format_input(chat_prompt)

    # Test with another invalid type
    chat_prompt = Chat(messages=[
        {"role": "user", "content": {"invalid": "dict"}}  # Invalid content type (dict not in list)
    ])

    with pytest.raises(ValueError, match="Invalid content type"):
        adapter.format_input(chat_prompt)


def test_transformers_multimodal_type_adapter_format_asset_for_template(adapter, image, video, audio):
    # Test Image asset
    image_asset = Image(image)
    formatted_image = adapter._format_asset_for_template(image_asset)
    assert formatted_image == {"type": "image", "image": image_asset}

    # Test Video asset
    video_asset = Video(video)
    formatted_video = adapter._format_asset_for_template(video_asset)
    assert formatted_video == {"type": "video", "video": video_asset}

    # Test Audio asset
    audio_asset = Audio(audio)
    formatted_audio = adapter._format_asset_for_template(audio_asset)
    assert formatted_audio == {"type": "audio", "audio": audio_asset}


def test_transformers_multimodal_type_adapter_format_asset_for_template_invalid_type(adapter):
    class MockUnsupportedAsset:
        pass

    # This test requires accessing the private method directly since the error
    # would normally be caught earlier in the validation chain
    unsupported_asset = MockUnsupportedAsset()

    with pytest.raises(ValueError, match="Assets must be of type `Image`, `Video` or `Audio`"):
        adapter._format_asset_for_template(unsupported_asset)


def test_transformers_multimodal_type_adapter_multiple_assets_in_single_item(adapter, image):
    image_asset = Image(image)
    video_asset = Video("dummy_video")

    chat_prompt = Chat(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this?"},
            {"type": "image", "image": image_asset, "video": video_asset}  # Multiple asset types
        ]}
    ])

    with pytest.raises(ValueError, match="Found item with multiple keys:"):
        adapter.format_input(chat_prompt)



def test_transformers_multimodal_type_adapter_correct_multiple_assets_usage(adapter, image):
    image_asset1 = Image(image)
    image_asset2 = Image(image)

    # Correct way: separate dictionary items for each asset
    chat_prompt = Chat(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What's in these images?"},
            {"type": "image", "image": image_asset1},
            {"type": "image", "image": image_asset2}
        ]}
    ])

    result = adapter.format_input(chat_prompt)
    assert isinstance(result, dict)
    assert "text" in result
    assert "images" in result
    assert len(result["images"]) == 2
