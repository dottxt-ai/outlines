import pytest
import outlines
import transformers
from outlines.types import Regex


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt2",
        "EleutherAI/pythia-70m-deduped",
    ],
)
def test_transformers_parametrized_smoke(model_name, hf_cache_dir):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=hf_cache_dir
    )
    hf_model.eval()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=hf_cache_dir
    )

    if hf_tokenizer.pad_token is None:
        assert hf_tokenizer.eos_token is not None
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        hf_model.config.pad_token_id = hf_model.config.eos_token_id

    model = outlines.from_transformers(hf_model, hf_tokenizer)

    prompt = "Is 1+1=2? Answer Yes or No:"
    constraint = Regex(r"\s*(Yes|No)")

    for _ in range(3):
        out = model(
            prompt,
            constraint,
            max_new_tokens=5,
            do_sample=False,
        )
        assert out.strip() in {"Yes", "No"}

