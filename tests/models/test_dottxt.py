import json
import os

import pytest
from dottxt.client import Dottxt as DottxtClient
from pydantic import BaseModel

import outlines
from outlines import Generator
from outlines.models.dottxt import Dottxt


MODEL_NAME = "dottxt/dottxt-v1-alpha"
MODEL_REVISION = "d06c86726aadd8dadb92c5b9b9e3ce8ef246c471"


class User(BaseModel):
    first_name: str
    last_name: str
    user_id: int


@pytest.fixture(scope="session")
def api_key():
    """Get the Dottxt API key from the environment, providing a default value
    if not found.

    This fixture should be used for tests that do not make actual api calls,
    but still require to initialize the Dottxt client.

    """
    api_key = os.getenv("DOTTXT_API_KEY")
    if not api_key:
        return "MOCK_API_KEY"
    return api_key


@pytest.fixture(scope="session")
def model_name_and_revision(api_key):
    client = DottxtClient(api_key=api_key)
    model_list = client.list_models()
    return (model_list[0].name, model_list[0].revision)


@pytest.fixture(scope="session")
def model(api_key, model_name_and_revision):
    client = DottxtClient(api_key=api_key)
    return Dottxt(
        client,
        model_name_and_revision[0],
        model_name_and_revision[1],
    )


@pytest.fixture(scope="session")
def model_no_model_name(api_key):
    client = DottxtClient(api_key=api_key)
    return Dottxt(client)


@pytest.mark.api_call
def test_dottxt_init_from_client(api_key, model_name_and_revision):
    client = DottxtClient(api_key=api_key)

    # Without model name
    model = outlines.from_dottxt(client)
    assert isinstance(model, Dottxt)
    assert model.client == client
    assert model.model_name is None

    # With model name
    model = outlines.from_dottxt(
        client,
        model_name_and_revision[0],
        model_name_and_revision[1],
    )
    assert isinstance(model, Dottxt)
    assert model.client == client
    assert model.model_name == model_name_and_revision[0]
    assert model.model_revision == model_name_and_revision[1]


def test_dottxt_wrong_output_type(model_no_model_name):
    with pytest.raises(TypeError, match="You must provide an output type"):
        model_no_model_name("prompt")


def test_dottxt_wrong_input_type(model_no_model_name):
    with pytest.raises(TypeError, match="is not available"):
        model_no_model_name(["prompt"], User)


@pytest.mark.api_call
def test_dottxt_wrong_inference_parameters(model_no_model_name):
    with pytest.raises(TypeError, match="got an unexpected"):
        model_no_model_name("prompt", User, foo=10)


@pytest.mark.api_call
def test_dottxt_direct_pydantic_call(model_no_model_name):
    result = model_no_model_name("Create a user", User)
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
def test_dottxt_direct_jsonschema_call(
    model_no_model_name, model_name_and_revision
):
    result = model_no_model_name(
        "Create a user",
        User,
        model_name=model_name_and_revision[0],
        model_revision=model_name_and_revision[1],
    )
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
def test_dottxt_generator_pydantic_call(model):
    generator = Generator(model, User)
    result = generator("Create a user")
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
def test_dottxt_streaming(model):
    with pytest.raises(
        NotImplementedError,
        match="Dottxt does not support streaming"
    ):
        model.stream("Create a user", User)
