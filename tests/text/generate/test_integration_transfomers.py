import numpy as np

import outlines.models as models
from outlines.text.generate.continuation import continuation


def test_transformers_integration_completion():
    rng = np.random.default_rng(0)

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = models.transformers(model_name, device="cpu")
    sequence = continuation(model)("prompt", rng=rng)
    assert isinstance(sequence, str)
    assert model.tokenizer.eos_token not in sequence

    sequence = continuation(model, max_tokens=10)("prompt", rng=rng)
    assert isinstance(sequence, str)


def test_transformers_integration_with_pad_token():
    model_name = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM"
    model = models.transformers(model_name, device="cpu")
    assert model.tokenizer.pad_token_id == 1
    assert model.tokenizer.pad_token == "<pad>"
