import pytest

from PIL import Image as PILImage
from outlines_core import Index, Vocabulary
from transformers import AutoTokenizer, LogitsProcessorList

from outlines.inputs import Chat, Image, Video
from outlines.models.transformers import TransformersMultiModalTypeAdapter
from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor


MODEL_NAME = "erwanf/gpt2-mini"


@pytest.fixture
def adapter():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    type_adapter = TransformersMultiModalTypeAdapter(tokenizer=tokenizer)
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
    image.format = "PNG"

    return image


def test_transformers_multimodal_type_adapter_format_input(adapter, image):
    with pytest.raises(TypeError):
        adapter.format_input("hello")

    with pytest.raises(ValueError):
        with pytest.deprecated_call():
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
