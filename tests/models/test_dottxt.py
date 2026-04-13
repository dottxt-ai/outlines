import json
import os

import pytest
from dottxt.client import AsyncDotTxt as AsyncDottxtClient
from dottxt.client import DotTxt as DottxtClient
from pydantic import BaseModel

import outlines
from outlines import Generator
from outlines.models.dottxt import AsyncDottxt, Dottxt


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
def model_name(api_key):
    client = DottxtClient(api_key=api_key)
    models = client.models.list()
    return models.data[0].id


@pytest.fixture(scope="session")
def model(api_key, model_name):
    client = DottxtClient(api_key=api_key)
    return Dottxt(client, model_name)


@pytest.fixture(scope="session")
def model_no_model_name(api_key):
    client = DottxtClient(api_key=api_key)
    return Dottxt(client)


@pytest.fixture(scope="session")
def async_model(api_key, model_name):
    client = AsyncDottxtClient(api_key=api_key)
    return AsyncDottxt(client, model_name)


@pytest.fixture(scope="session")
def async_model_no_model_name(api_key):
    client = AsyncDottxtClient(api_key=api_key)
    return AsyncDottxt(client)


# ── Sync tests ────────────────────────────────────────────────────────────────

@pytest.mark.api_call
def test_dottxt_init_from_client(api_key, model_name):
    client = DottxtClient(api_key=api_key)

    model = outlines.from_dottxt(client)
    assert isinstance(model, Dottxt)
    assert model.client == client
    assert model.model is None

    model = outlines.from_dottxt(client, model_name)
    assert isinstance(model, Dottxt)
    assert model.client == client
    assert model.model == model_name


def test_dottxt_missing_model(model_no_model_name):
    with pytest.raises(ValueError, match="A model identifier is required"):
        model_no_model_name("prompt", User)


def test_dottxt_wrong_output_type(model_no_model_name):
    with pytest.raises(TypeError, match="You must provide an output type"):
        model_no_model_name("prompt")


def test_dottxt_wrong_input_type(model_no_model_name):
    with pytest.raises(TypeError, match="is not available"):
        model_no_model_name(["prompt"], User)


@pytest.mark.api_call
def test_dottxt_wrong_inference_parameters(model):
    with pytest.raises(TypeError, match="got an unexpected"):
        model("prompt", User, foo=10)


@pytest.mark.api_call
def test_dottxt_direct_pydantic_call(model):
    result = model("Create a user", User)
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
def test_dottxt_direct_call_with_model(model_no_model_name, model_name):
    result = model_no_model_name(
        "Create a user",
        User,
        model=model_name,
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


@pytest.mark.api_call
def test_dottxt_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."]
        )


# ── Async tests ───────────────────────────────────────────────────────────────

@pytest.mark.api_call
def test_dottxt_async_init_from_client(api_key, model_name):
    client = AsyncDottxtClient(api_key=api_key)

    model = outlines.from_dottxt(client)
    assert isinstance(model, AsyncDottxt)
    assert model.client == client
    assert model.model is None

    model = outlines.from_dottxt(client, model_name)
    assert isinstance(model, AsyncDottxt)
    assert model.client == client
    assert model.model == model_name


@pytest.mark.asyncio
async def test_dottxt_async_missing_model(async_model_no_model_name):
    with pytest.raises(ValueError, match="A model identifier is required"):
        await async_model_no_model_name("prompt", User)


def test_dottxt_async_wrong_output_type(async_model_no_model_name):
    with pytest.raises(TypeError, match="You must provide an output type"):
        # format_output_type is sync and raises before the coroutine is awaited
        async_model_no_model_name.type_adapter.format_output_type(None)


def test_dottxt_async_wrong_input_type(async_model_no_model_name):
    with pytest.raises(TypeError, match="is not available"):
        async_model_no_model_name.type_adapter.format_input(["prompt"])


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_dottxt_async_direct_pydantic_call(async_model):
    result = await async_model("Create a user", User)
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_dottxt_async_direct_call_with_model(
    async_model_no_model_name, model_name
):
    result = await async_model_no_model_name(
        "Create a user",
        User,
        model=model_name,
    )
    assert "first_name" in json.loads(result)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_dottxt_async_streaming(async_model):
    with pytest.raises(
        NotImplementedError,
        match="Dottxt does not support streaming"
    ):
        async for _ in async_model.stream("Create a user", User):
            pass


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_dottxt_async_batch(async_model):
    with pytest.raises(NotImplementedError, match="does not support"):
        await async_model.batch(
            ["Respond with one word.", "Respond with one word."]
        )
