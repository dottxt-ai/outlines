import pytest

import outlines.models as models
import outlines.models.routers as routers


def test_text_model_invalid_provider():
    with pytest.raises(ValueError, match="model provider"):
        routers.text_completion("xx/model_name")


def test_text_model_router():
    dummy_model_name = "model_name"
    llm_builder = routers.text_completion(f"openai/{dummy_model_name}")
    assert llm_builder.func == models.OpenAICompletion
    assert llm_builder.args == (dummy_model_name,)

    llm_builder = routers.text_completion(f"hf/{dummy_model_name}")
    assert llm_builder.func == models.HuggingFaceCompletion
    assert llm_builder.args == (dummy_model_name,)


def test_invalid_model_path():
    with pytest.raises(ValueError, match="must be in the form"):
        routers.parse_model_path("hf")
