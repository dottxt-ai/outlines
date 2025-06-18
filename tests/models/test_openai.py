import importlib
import json
from contextlib import contextmanager
from unittest import mock
from unittest.mock import MagicMock

import pytest
from openai import AsyncOpenAI

from outlines import generate
from outlines.models.openai import OpenAI, OpenAIConfig


def module_patch(path):
    """Patch functions that have the same name as the module in which they're implemented."""
    target = path
    components = target.split(".")
    for i in range(len(components), 0, -1):
        try:
            # attempt to import the module
            imported = importlib.import_module(".".join(components[:i]))

            # module was imported, let's use it in the patch
            patch = mock.patch(path)
            patch.getter = lambda: imported
            patch.attribute = ".".join(components[i:])
            return patch
        except Exception:
            continue

    # did not find a module, just return the default mock
    return mock.patch(path)


def test_openai_call():
    with module_patch("outlines.models.openai.generate_chat") as mocked_generate_chat:
        mocked_generate_chat.return_value = ["foo"], 1, 2
        async_client = MagicMock(spec=AsyncOpenAI, api_key="key")

        model = OpenAI(
            async_client,
            OpenAIConfig(max_tokens=10, temperature=0.5, n=2, stop=["."]),
        )

        assert model("bar")[0] == "foo"
        assert model.prompt_tokens == 1
        assert model.completion_tokens == 2
        mocked_generate_chat_args = mocked_generate_chat.call_args
        mocked_generate_chat_arg_config = mocked_generate_chat_args[0][3]
        assert isinstance(mocked_generate_chat_arg_config, OpenAIConfig)
        assert mocked_generate_chat_arg_config.max_tokens == 10
        assert mocked_generate_chat_arg_config.temperature == 0.5
        assert mocked_generate_chat_arg_config.n == 2
        assert mocked_generate_chat_arg_config.stop == ["."]

        model("bar", samples=3)
        mocked_generate_chat_args = mocked_generate_chat.call_args
        mocked_generate_chat_arg_config = mocked_generate_chat_args[0][3]
        assert mocked_generate_chat_arg_config.n == 3


@contextmanager
def patched_openai(completion, **oai_config):
    """Create a patched openai whose chat completions always returns `completion`"""
    with module_patch("outlines.models.openai.generate_chat") as mocked_generate_chat:
        mocked_generate_chat.return_value = completion, 1, 2
        async_client = MagicMock(spec=AsyncOpenAI, api_key="key")
        model = OpenAI(
            async_client,
            OpenAIConfig(max_tokens=10, temperature=0.5, n=2, stop=["."]),
        )
        yield model


def test_openai_choice_call():
    with patched_openai(completion='{"result": "foo"}') as model:
        generator = generate.choice(model, ["foo", "bar"])
        assert generator("hi") == "foo"


def test_openai_choice_call_invalid_server_response():
    with patched_openai(completion="not actual json") as model:
        generator = generate.choice(model, ["foo", "bar"])
        with pytest.raises(json.decoder.JSONDecodeError):
            generator("hi")


def test_openai_json_call_pydantic():
    from pydantic import BaseModel, ConfigDict, ValidationError

    class Person(BaseModel):
        model_config = ConfigDict(extra="forbid")  # required for openai
        first_name: str
        last_name: str
        age: int

    completion = '{"first_name": "Usain", "last_name": "Bolt", "age": 38}'

    # assert success for valid response
    with patched_openai(completion=completion) as model:
        generator = generate.json(model, Person)
        assert generator("fastest person") == Person.parse_raw(completion)

    # assert fail for non-json response
    with patched_openai(completion="usain bolt") as model:
        generator = generate.json(model, Person)
        with pytest.raises(ValidationError):
            assert generator("fastest person")


def test_openai_json_call_str():
    person_schema = '{"additionalProperties": false, "properties": {"first_name": {"title": "First Name", "type": "string"}, "last_name": {"title": "Last Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}}, "required": ["first_name", "last_name", "age"], "title": "Person", "type": "object"}'

    output = {"first_name": "Usain", "last_name": "Bolt", "age": 38}

    # assert success for valid response
    with patched_openai(completion=json.dumps(output)) as model:
        generator = generate.json(model, person_schema)
        assert generator("fastest person") == output

    # assert fail for non-json response
    with patched_openai(completion="usain bolt") as model:
        generator = generate.json(model, person_schema)
        with pytest.raises(json.decoder.JSONDecodeError):
            assert generator("fastest person")
