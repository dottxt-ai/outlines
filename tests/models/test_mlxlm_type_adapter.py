import io
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from outlines_core import Index, Vocabulary
from PIL import Image as PILImage

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor
from outlines.inputs import Chat, Image
from outlines.models.mlxlm import (
    MLXLMMultiModalTypeAdapter,
    MLXLMTypeAdapter,
)

try:
    import mlx_lm
    import mlx.core as mx

    HAS_MLX = mx.metal.is_available()
except ImportError:
    HAS_MLX = False


MODEL_NAME = "mlx-community/SmolLM-135M-Instruct-4bit"


@pytest.fixture
def adapter():
    _, tokenizer = mlx_lm.load(MODEL_NAME)
    return MLXLMTypeAdapter(tokenizer=tokenizer)


@pytest.fixture
def logits_processor():
    vocabulary = Vocabulary.from_pretrained(MODEL_NAME)
    index = Index(r"[0-9]{3}", vocabulary)
    return OutlinesCoreLogitsProcessor(index, "mlx")


@pytest.fixture
def image():
    width, height = 1, 1
    white_background = (255, 255, 255)
    image = PILImage.new("RGB", (width, height), white_background)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)

    return image


def test_mlxlm_type_adapter_format_input_with_template():
    tokenizer = MagicMock()
    tokenizer.chat_template = "some_template"
    tokenizer.apply_chat_template.return_value = "formatted_prompt"

    adapter = MLXLMTypeAdapter(tokenizer=tokenizer, has_chat_template=True)
    message = "prompt"
    result = adapter.format_input(message)

    assert result == "formatted_prompt"
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "prompt"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def test_mlxlm_type_adapter_format_input_without_template():
    tokenizer = MagicMock()
    tokenizer.chat_template = None

    adapter = MLXLMTypeAdapter(tokenizer=tokenizer, has_chat_template=False)
    message = "prompt"
    result = adapter.format_input(message)

    assert result == "prompt"


@pytest.mark.parametrize(
    ("model_type", "formatted_prompt"),
    [
        ("gemma3", "Describe this.<start_of_image>"),
        ("phi3_v", "<|image_1|>Describe this."),
        ("paligemma", "<image>Describe this."),
    ],
)
def test_mlxlm_multimodal_type_adapter_uses_model_prompt_formatter(
    monkeypatch, image, model_type, formatted_prompt
):
    processor = MagicMock()
    config = SimpleNamespace(model_type=model_type)
    formatter = MagicMock(return_value=formatted_prompt)
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.apply_chat_template = formatter
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)

    adapter = MLXLMMultiModalTypeAdapter(
        processor=processor,
        config=config,
    )
    image_asset = Image(image)
    result = adapter.format_input(Chat([{
        "role": "user",
        "content": ["Describe this.", image_asset],
    }]))

    assert result == {
        "prompt": formatted_prompt,
        "images": [image_asset.image],
    }
    formatter.assert_called_once_with(
        processor,
        config,
        [{"role": "user", "content": "Describe this."}],
        num_images=1,
    )


def test_mlxlm_multimodal_type_adapter_rejects_output_type():
    adapter = MLXLMMultiModalTypeAdapter(
        processor=MagicMock(),
        config=SimpleNamespace(model_type="smolvlm"),
    )

    assert adapter.format_output_type(None) is None
    with pytest.raises(NotImplementedError, match="structured generation"):
        adapter.format_output_type(MagicMock())


def test_mlxlm_multimodal_type_adapter_rejects_later_turn_images(
    monkeypatch, image
):
    formatter = MagicMock(return_value="formatted_prompt")
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.apply_chat_template = formatter
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    adapter = MLXLMMultiModalTypeAdapter(
        processor=MagicMock(),
        config=SimpleNamespace(model_type="smolvlm"),
    )

    with pytest.raises(ValueError, match="first user message"):
        adapter.format_input(Chat([
            {"role": "user", "content": "Remember this."},
            {"role": "assistant", "content": "Okay."},
            {
                "role": "user",
                "content": ["Describe this.", Image(image)],
            },
        ]))

    formatter.assert_not_called()


def test_mlxlm_multimodal_type_adapter_rejects_images_after_two_messages(
    monkeypatch, image
):
    formatter = MagicMock(return_value="formatted_prompt")
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.apply_chat_template = formatter
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    adapter = MLXLMMultiModalTypeAdapter(
        processor=MagicMock(),
        config=SimpleNamespace(model_type="smolvlm"),
    )

    with pytest.raises(ValueError, match="first or second message"):
        adapter.format_input(Chat([
            {"role": "system", "content": "First instruction."},
            {"role": "system", "content": "Second instruction."},
            {
                "role": "user",
                "content": ["Describe this.", Image(image)],
            },
        ]))

    formatter.assert_not_called()


def test_mlxlm_multimodal_type_adapter_rejects_two_leading_user_messages(
    monkeypatch, image
):
    formatter = MagicMock(return_value="formatted_prompt")
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.apply_chat_template = formatter
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    adapter = MLXLMMultiModalTypeAdapter(
        processor=MagicMock(),
        config=SimpleNamespace(model_type="smolvlm"),
    )

    with pytest.raises(ValueError, match="first two messages"):
        adapter.format_input(Chat([
            {
                "role": "user",
                "content": ["Describe this.", Image(image)],
            },
            {"role": "user", "content": "Focus on the background."},
        ]))

    formatter.assert_not_called()


@pytest.mark.parametrize("model_type", ["paligemma", "molmo", "florence2"])
def test_mlxlm_multimodal_type_adapter_rejects_prompt_only_chat_history(
    monkeypatch, image, model_type
):
    formatter = MagicMock(return_value="formatted_prompt")
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.apply_chat_template = formatter
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    adapter = MLXLMMultiModalTypeAdapter(
        processor=MagicMock(),
        config=SimpleNamespace(model_type=model_type),
    )

    with pytest.raises(ValueError, match="single user message"):
        adapter.format_input(Chat([
            {
                "role": "user",
                "content": ["Describe this.", Image(image)],
            },
            {"role": "assistant", "content": "Prior answer."},
        ]))

    formatter.assert_not_called()


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_type_adapter_format_input(adapter, image):
    # Anything else than a string/Chat (invalid)
    with pytest.raises(NotImplementedError):
        adapter.format_input(["Hello, world!"])

    # String
    assert adapter.format_input("Hello, world!") == "Hello, world!"

    # Chat
    messages = [
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hello, world!"},
    ]
    expected = (
        "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n"
        + "Hello, world!<|im_end|>\n<|im_start|>assistant\n"
    )
    assert adapter.format_input(Chat(messages=messages)) == expected

    # Multi-modal (invalid)
    with pytest.raises(
        ValueError,
        match="mlx-lm does not support multi-modal messages."
    ):
        adapter.format_input(Chat(messages=[
            {"role": "user", "content": ["prompt", Image(image)]},
        ]))


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_type_adapter_format_output_type(adapter, logits_processor):
    formatted = adapter.format_output_type(logits_processor)
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert isinstance(formatted[0], OutlinesCoreLogitsProcessor)
