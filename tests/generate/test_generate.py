import re

import pytest

import outlines.generate as generate
import outlines.models as models


@pytest.fixture(scope="session")
def model_llamacpp(tmp_path_factory):
    return models.llamacpp(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )


@pytest.fixture(scope="session")
def model_mlxlm(tmp_path_factory):
    return models.mlxlm("mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")


@pytest.fixture(scope="session")
def model_transformers(tmp_path_factory):
    return models.transformers("Locutusque/TinyMistral-248M-v2-Instruct", device="cpu")


@pytest.mark.parametrize(
    "model_fixture",
    ("model_llamacpp", "model_mlxlm", "model_transformers"),
)
def test_generate_text(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model)
    res = generator("test", max_tokens=10)
    assert isinstance(res, str)


@pytest.mark.parametrize(
    "model_fixture",
    ("model_llamacpp", "model_mlxlm", "model_transformers"),
)
@pytest.mark.parametrize(
    "pattern",
    (
        "[0-9]",
        "abc*",
        "\\+?[1-9][0-9]{7,14}",
    ),
)
def test_generate_regex(request, model_fixture, pattern):
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    res = generator("foobarbaz", max_tokens=20)
    assert re.match(pattern, res) is not None, res
