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

    tokenizer = TransformersTokenizer(
        TEST_MODEL, additional_special_tokens=["<t1>", "<t2>"]
    )
    assert "<t1>" in tokenizer.special_tokens
    assert "<t2>" in tokenizer.special_tokens


def test_llama_tokenizer():
    tokenizer = TransformersTokenizer("hf-internal-testing/llama-tokenizer")

    # Broken
    assert tokenizer.tokenizer.convert_tokens_to_string(["▁baz"]) == "baz"
    assert tokenizer.tokenizer.convert_tokens_to_string(["<0x20>"]) == ""
    assert tokenizer.tokenizer.convert_tokens_to_string(["▁▁▁"]) == "  "

    # Not broken
    assert tokenizer.convert_token_to_string("▁baz") == " baz"
    assert tokenizer.convert_token_to_string("<0x20>") == " "
    assert tokenizer.convert_token_to_string("▁▁▁") == "   "


def test_model():
    with pytest.raises(ValueError, match="When passing device_map as a string"):
        transformers(TEST_MODEL, device="non_existent")

    model = transformers(TEST_MODEL, device="cpu")
    assert isinstance(model.tokenizer, TransformersTokenizer)
    assert model.device.type == "cpu"

    model = transformers(TEST_MODEL, model_kwargs={"device_map": "cpu"})
    assert isinstance(model.tokenizer, TransformersTokenizer)
    assert model.device.type == "cpu"

    model = transformers(TEST_MODEL, device="cpu", model_kwargs={"device_map": "cuda"})
    assert isinstance(model.tokenizer, TransformersTokenizer)
    assert model.device.type == "cpu"

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

    with pytest.raises(AssertionError):
        input_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [0, 1, 2]]])
        logits = model(input_ids, torch.ones_like(input_ids))


def test_tokenizer_eq_hash():
    tokenizer = TransformersTokenizer("gpt2")
    tokenizer2 = TransformersTokenizer("gpt2")
    assert tokenizer == tokenizer2
    assert hash(tokenizer) == hash(tokenizer2)
