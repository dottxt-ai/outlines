import pytest
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


@pytest.fixture(scope="session")
def hf_cache_dir():
    """Shared Hugging Face cache directory for all model downloads."""
    path = Path(".hf_cache")
    path.mkdir(exist_ok=True)
    return str(path)


@pytest.mark.parametrize(
    "backend, model_name",
    [
        ("transformers", "sshleifer/tiny-gpt2"),
        ("transformers", "distilgpt2"),
        pytest.param("vllm", "facebook/opt-125m", marks=pytest.mark.skipif(True, reason="vLLM not available on Windows")),
    ],
)
@pytest.mark.slow
@pytest.mark.flaky(reruns=2)
def test_parametrized_steerable_model(backend, model_name, hf_cache_dir):
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=hf_cache_dir)
    tokenizer = Au
    toTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)

    model = outlines.models.transformers.Transformers(hf_model, tokenizer)

    prompt = "Hello world"
    outputs = set()

    for _ in range(3):
        output = model.generate(prompt, max_new_tokens=10)
        assert isinstance(output, str)
        assert len(output.strip()) > 0
        outputs.add(output)

    assert len(outputs) >= 1
