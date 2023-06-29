import pytest
import torch
from transformers.models.gpt2 import GPT2TokenizerFast

from outlines.models.transformers import TransformersTokenizer, transformers

TEST_MODEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"


def test_tokenizer():
    tokenizer = TransformersTokenizer(TEST_MODEL)
    assert tokenizer.eos_token_id == 0
    assert tokenizer.pad_token_id == 0
    assert isinstance(tokenizer.tokenizer, GPT2TokenizerFast)

    token_ids, attention_mask = tokenizer.encode("Test")
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 1
    assert isinstance(token_ids, torch.LongTensor)
    assert token_ids.shape == attention_mask.shape

    token_ids, attention_mask = tokenizer.encode(["Test", "Test"])
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 2
    assert isinstance(token_ids, torch.LongTensor)
    assert token_ids.shape == attention_mask.shape

    token_ids, attention_mask = tokenizer.encode(["Test", "A long sentence"])
    assert token_ids.shape == attention_mask.shape
    assert attention_mask[0][0] == tokenizer.pad_token_id

    text = tokenizer.decode(torch.tensor([[0, 1, 2]]))
    isinstance(text, str)

    text = tokenizer.decode(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    isinstance(text, list)
    isinstance(text[0], str)
    isinstance(text[1], str)


def test_model():
    with pytest.raises(RuntimeError, match="Expected one of cpu, cuda"):
        transformers(TEST_MODEL, device="non_existent")

    model = transformers(TEST_MODEL, device="cpu")
    assert isinstance(model.tokenizer, TransformersTokenizer)
    assert model.device == "cpu"

    input_ids = torch.tensor([[0, 1, 2]])
    logits = model(input_ids, torch.ones_like(input_ids))
    assert logits.type() == "torch.FloatTensor"
    assert logits.ndim == 2
    assert logits.shape[0] == 1

    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    logits = model(input_ids, torch.ones_like(input_ids))
    assert logits.type() == "torch.FloatTensor"
    assert logits.ndim == 2
    assert logits.shape[0] == 3

    input_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [0, 1, 2]]])
    logits = model(input_ids, torch.ones_like(input_ids))
    assert logits.ndim == 3
    assert logits.shape[0] == 2
    assert logits.shape[1] == 2
    assert torch.equal(logits[0][0], logits[1][1])
