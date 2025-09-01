import pytest
import io

from outlines_core import Index, Vocabulary
from PIL import Image as PILImage

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor
from outlines.inputs import Chat, Image
from outlines.models.mlxlm import MLXLMTypeAdapter

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
