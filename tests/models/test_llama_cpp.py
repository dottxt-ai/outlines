import numpy as np
import pytest
import torch
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


def test_tokenizer(model_download):
    model = llamacpp(TEST_MODEL, "cpu")

    tokenizer = model.tokenizer
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == -1

    token_ids, attention_mask = tokenizer.encode(["Test", "test bla hallo"])
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 2
    assert token_ids[0, -1] == -1
    assert token_ids[1, -1] != -1
    assert isinstance(token_ids, torch.LongTensor)
    assert token_ids.shape == attention_mask.shape

    token_ids, attention_mask = tokenizer.encode("Test")
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 1
    assert isinstance(token_ids, torch.LongTensor)
    assert token_ids.shape == attention_mask.shape

    text = tokenizer.decode(np.array([0, 1, 2]))
    assert isinstance(text, list)


def test_model(model_download):
    model = llamacpp(TEST_MODEL)

    input_ids = torch.tensor([[0, 1, 2]])
    logits, kv_cache = model(input_ids, torch.ones_like(input_ids))

    assert logits.ndim == 2
    assert logits.shape[0] == 1

    model.n_past = 0
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    logits, kv_cache = model(input_ids, torch.ones_like(input_ids))

    assert logits.ndim == 2
    assert logits.shape[0] == 3

    model.n_past = 0
    input_ids = torch.tensor([[0, 1, 2], [3, -1, -1]])
    logits, kv_cache = model(input_ids, torch.ones_like(input_ids))

    assert logits.ndim == 2
    assert logits.shape[0] == 2
