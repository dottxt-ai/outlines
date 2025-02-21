import json
import os

import pytest
from pydantic import BaseModel

from outlines.generate import Generator
from outlines.models.dottxt import Dottxt
from outlines.types import Json


class User(BaseModel):
    first_name: str
    last_name: str
    user_id: int


@pytest.fixture
def api_key():
    """Get the Dottxt API key from the environment, providing a default value if not found.

    This fixture should be used for tests that do not make actual api calls,
    but still require to initialize the Dottxt client.

    """
    api_key = os.getenv("DOTTXT_API_KEY")
    if not api_key:
        return "MOCK_API_KEY"
    return api_key


def test_dottxt_wrong_init_parameters(api_key):
    with pytest.raises(TypeError, match="got an unexpected"):
        Dottxt(api_key=api_key, foo=10)


def test_dottxt_wrong_output_type(api_key):
    with pytest.raises(NotImplementedError, match="must provide an output type"):
        model = Dottxt(api_key=api_key)
        model("prompt")


@pytest.mark.api_call
def test_dottxt_wrong_input_type(api_key):
    with pytest.raises(NotImplementedError, match="is not available"):
        model = Dottxt(api_key=api_key)
        model(["prompt"], Json(User))


@pytest.mark.api_call
def test_dottxt_wrong_inference_parameters(api_key):
    with pytest.raises(TypeError, match="got an unexpected"):
        model = Dottxt(api_key=api_key)
        model("prompt", Json(User), foo=10)


@pytest.mark.api_call
def test_dottxt_direct_call(api_key):
    model = Dottxt(api_key=api_key, model_name="meta-llama/Llama-3.1-8B-Instruct")
    result = model("Create a user", Json(User))
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
def test_dottxt_generator_call(api_key):
    model = Dottxt(api_key=api_key, model_name="meta-llama/Llama-3.1-8B-Instruct")
    generator = Generator(model, Json(User))
    result = generator("Create a user")
    assert "first_name" in json.loads(result)
