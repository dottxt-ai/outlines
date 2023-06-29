import torch

import outlines.models as models
from outlines.text.generate.continuation import continuation


def test_transformers_integration_continuation():
    rng = torch.Generator()
    rng.manual_seed(10000)  # Choosen so <EOS> is generated

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    sequence = continuation(model)("Write a short sentence", rng=rng)
    assert isinstance(sequence, str)
    assert model.tokenizer.eos_token not in sequence

    sequence = continuation(model, max_tokens=10)("Write a short sentence", rng=rng)
    assert isinstance(sequence, str)


def test_transformers_integration_with_pad_token():
    model_name = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM"
    model = models.transformers(model_name, device="cpu")
    assert model.tokenizer.pad_token_id == 1
    assert model.tokenizer.pad_token == "<pad>"
