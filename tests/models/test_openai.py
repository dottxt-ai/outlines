import importlib
from unittest import mock
from unittest.mock import MagicMock

import pytest
from openai import AsyncOpenAI

from outlines.models.openai import (
    OpenAI,
    OpenAIConfig,
    build_optimistic_mask,
    find_longest_intersection,
    find_response_choices_intersection,
)


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


@pytest.mark.parametrize(
    "response,choice,expected_intersection,expected_choices_left",
    (
        ([1, 2, 3, 4], [[5, 6]], [], [[5, 6]]),
        ([1, 2, 3, 4], [[5, 6], [7, 8]], [], [[5, 6], [7, 8]]),
        ([1, 2, 3, 4], [[1, 2], [7, 8]], [1, 2], [[]]),
        ([1, 2], [[1, 2, 3, 4], [1, 2]], [1, 2], [[3, 4], []]),
        ([1, 2, 3], [[1, 2, 3, 4], [1, 2]], [1, 2, 3], [[4]]),
    ),
)
def test_find_response_choices_intersection(
    response, choice, expected_intersection, expected_choices_left
):
    intersection, choices_left = find_response_choices_intersection(response, choice)
    assert intersection == expected_intersection
    assert choices_left == expected_choices_left


@pytest.mark.parametrize(
    "response,choice,expected_prefix",
    (
        ([1, 2, 3], [1, 2, 3, 4], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3], [1, 2, 3]),
        ([4, 5], [1, 2, 3], []),
    ),
)
def test_find_longest_common_prefix(response, choice, expected_prefix):
    prefix = find_longest_intersection(response, choice)
    assert prefix == expected_prefix


@pytest.mark.parametrize(
    "transposed,mask_size,expected_mask",
    (
        ([{1, 2}, {3, 4}], 3, {1: 100, 2: 100, 3: 100}),
        ([{1, 2}, {3, 4}], 4, {1: 100, 2: 100, 3: 100, 4: 100}),
    ),
)
def test_build_optimistic_mask(transposed, mask_size, expected_mask):
    mask = build_optimistic_mask(transposed, mask_size)
    assert mask == expected_mask
