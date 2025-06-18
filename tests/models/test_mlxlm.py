import pytest

from outlines.models.mlxlm import mlxlm
from outlines.models.transformers import TransformerTokenizer

try:
    import mlx.core as mx

    HAS_MLX = mx.metal.is_available()
except ImportError:
    HAS_MLX = False


TEST_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_model():
    model = mlxlm(TEST_MODEL)
    assert hasattr(model, "model")
    assert hasattr(model, "tokenizer")
    assert isinstance(model.tokenizer, TransformerTokenizer)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_tokenizer():
    model = mlxlm(TEST_MODEL)

    # Test single string encoding/decoding
    test_text = "Hello, world!"
    token_ids = mx.array(model.mlx_tokenizer.encode(test_text))
    assert isinstance(token_ids, mx.array)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_generate():
    from outlines.generate.api import GenerationParameters, SamplingParameters

    model = mlxlm(TEST_MODEL)
    prompt = "Write a haiku about programming:"

    # Test with basic generation parameters
    gen_params = GenerationParameters(max_tokens=50, stop_at=None, seed=None)

    # Test with different sampling parameters
    sampling_params = SamplingParameters(
        sampler="multinomial", num_samples=1, top_p=0.9, top_k=None, temperature=0.7
    )

    # Test generation
    output = model.generate(prompt, gen_params, None, sampling_params)
    assert isinstance(output, str)
    assert len(output) > 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_stream():
    from outlines.generate.api import GenerationParameters, SamplingParameters

    model = mlxlm(TEST_MODEL)
    prompt = "Count from 1 to 5:"

    gen_params = GenerationParameters(max_tokens=20, stop_at=None, seed=None)

    sampling_params = SamplingParameters(
        sampler="greedy",  # Use greedy sampling for deterministic output
        num_samples=1,
        top_p=None,
        top_k=None,
        temperature=0.0,
    )

    # Test streaming
    stream = model.stream(prompt, gen_params, None, sampling_params)
    tokens = list(stream)
    assert len(tokens) > 0
    assert all(isinstance(token, str) for token in tokens)

    # Test that concatenated streaming output matches generate output
    streamed_text = "".join(tokens)
    generated_text = model.generate(prompt, gen_params, None, sampling_params)
    assert streamed_text == generated_text


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_errors():
    model = mlxlm(TEST_MODEL)

    # Test batch inference (should raise NotImplementedError)
    with pytest.raises(NotImplementedError):
        from outlines.generate.api import GenerationParameters, SamplingParameters

        gen_params = GenerationParameters(max_tokens=10, stop_at=None, seed=None)
        sampling_params = SamplingParameters("multinomial", 1, None, None, 1.0)
        model.generate(["prompt1", "prompt2"], gen_params, None, sampling_params)

    # Test beam search (should raise NotImplementedError)
    with pytest.raises(NotImplementedError):
        sampling_params = SamplingParameters("beam_search", 1, None, None, 1.0)
        model.generate("test prompt", gen_params, None, sampling_params)
