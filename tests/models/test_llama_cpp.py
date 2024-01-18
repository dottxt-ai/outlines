import pytest
from huggingface_hub import hf_hub_download

from outlines.models.llamacpp import llamacpp

TEST_MODEL = "./llama-test-model/llama-2-tiny-random.gguf"


@pytest.fixture(scope="session")
def model_download(tmp_path_factory):
    tmp_path_factory.mktemp("./llama-test-model")
    hf_hub_download(
        repo_id="aladar/llama-2-tiny-random-GGUF",
        local_dir="./llama-test-model",
        local_dir_use_symlinks=False,
        filename="llama-2-tiny-random.gguf",
    )


def test_model(model_download):
    model = llamacpp(TEST_MODEL)

    completion = model("Some string")

    assert isinstance(completion, str)

    model.model.reset()
    completions = model(["Some string", "Other string"])

    assert isinstance(completions, list)
    assert len(completions) == 2
    assert isinstance(completions[0], str)
    assert isinstance(completions[1], str)
