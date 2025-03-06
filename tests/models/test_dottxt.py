import json
import os

import pytest
from dottxt.client import Dottxt as DottxtClient
from pydantic import BaseModel

import outlines
from outlines import Generator
from outlines.models.dottxt import Dottxt


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
        client = DottxtClient(api_key=api_key)
        Dottxt(client, foo=10)


def test_dottxt_wrong_output_type(api_key):
    with pytest.raises(TypeError, match="You must provide an output type"):
        client = DottxtClient(api_key=api_key)
        model = Dottxt(client)
        model("prompt")


def test_dottxt_init_from_client(api_key):
    client = DottxtClient(api_key=api_key)
    model = outlines.from_dottxt(client)
    assert isinstance(model, Dottxt)
    assert model.client == client


@pytest.mark.api_call
def test_dottxt_wrong_input_type(api_key):
    with pytest.raises(TypeError, match="is not available"):
        client = DottxtClient(api_key=api_key)
        model = Dottxt(client)
        model(["prompt"], User)


@pytest.mark.api_call
def test_dottxt_wrong_inference_parameters(api_key):
    with pytest.raises(TypeError, match="got an unexpected"):
        client = DottxtClient(api_key=api_key)
        model = Dottxt(client)
        model("prompt", User, foo=10)


@pytest.mark.api_call
def test_dottxt_direct_pydantic_call(api_key):
    client = DottxtClient(api_key=api_key)
    model = Dottxt(client)
    result = model("Create a user", User)
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
def test_dottxt_direct_jsonschema_call(api_key):
    client = DottxtClient(api_key=api_key)
    model = Dottxt(client)
    result = model("Create a user", User)
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
def test_dottxt_generator_pydantic_call(api_key):
    client = DottxtClient(api_key=api_key)
    model = Dottxt(client)
    generator = Generator(model, User)
    result = generator("Create a user")
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
def test_dottxt_streaming(api_key):
    client = DottxtClient(api_key=api_key)
    model = Dottxt(client)
    with pytest.raises(NotImplementedError, match="Dottxt does not support streaming"):
        model.stream("Create a user", User)
