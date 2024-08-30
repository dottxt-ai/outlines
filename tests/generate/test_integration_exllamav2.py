import importlib
from unittest.mock import patch

import pytest

import outlines.models as models
from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.models.exllamav2 import ExLlamaV2Model
from outlines.models.transformers import TransformerTokenizer


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    return models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        cache_q4=True,
        paged=False,
    )


def test_exl2_import_error():
    with patch.dict("sys.modules", {"exllamav2": None}):
        with pytest.raises(ImportError):
            models.exl2(
                model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
                cache_q4=True,
                paged=False,
            )


def test_model_attributes(model):
    assert hasattr(model, "generator")
    assert hasattr(model, "tokenizer")
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert hasattr(model, "max_seq_len")
    assert isinstance(model.max_seq_len, int)


def test_model_generate_prompt_types(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=10, stop_at=None, seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        1,
        0.9,
        50,
        1.0,
    )
    output = model.generate(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    assert isinstance(output, str)
    prompt = ["test"]
    output = model.generate(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    assert isinstance(output, str)


def test_model_generate_no_max_tokens(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=None, stop_at=None, seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        1,
        0.9,
        50,
        1.0,
    )
    output = model.generate(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    assert isinstance(output, str)


def test_model_generate_test_stop_at(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=10, stop_at="stop", seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        1,
        0.9,
        50,
        1.0,
    )
    output = model.generate(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    assert isinstance(output, str)
    generation_params = GenerationParameters(max_tokens=10, stop_at=["stop"], seed=None)
    output = model.generate(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    assert isinstance(output, str)


def test_model_generate_multisampling(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=10, stop_at="stop", seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        2,
    )
    output = model.generate(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    assert isinstance(output, list)
    assert isinstance(output[0], str)


def test_model_prepare_generation_parameters(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=10, stop_at="stop", seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        2,
    )
    exllamav2_params, prompts = model.prepare_generation_parameters(
        prompt, generation_params, sampling_params, structure_logits_processor
    )
    assert isinstance(exllamav2_params, dict)
    assert isinstance(prompts, list)


def test_model_stream_prompt_types(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=10, stop_at=None, seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        1,
        0.9,
        50,
        1.0,
    )
    generator = model.stream(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    for token in generator:
        assert isinstance(token, str)
    prompt = ["test"]
    generator = model.stream(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    for token in generator:
        assert isinstance(token, str)


def test_model_stream_no_max_tokens(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=None, stop_at=None, seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        1,
        0.9,
        50,
        1.0,
    )
    generator = model.stream(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    for token in generator:
        assert isinstance(token, str)


def test_model_stream_test_stop_at(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=10, stop_at="stop", seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        1,
        0.9,
        50,
        1.0,
    )
    generator = model.stream(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    for token in generator:
        assert isinstance(token, str)
    generation_params = GenerationParameters(max_tokens=10, stop_at=["stop"], seed=None)
    generator = model.stream(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    for token in generator:
        assert isinstance(token, str)


def test_model_stream_multisampling(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=10, stop_at="stop", seed=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        2,
    )
    generator = model.stream(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    for token in generator:
        assert isinstance(token, list)
        assert isinstance(token[0], str)


def test_model_stream_seed(model):
    prompt = "test"
    generation_params = GenerationParameters(max_tokens=10, seed=1, stop_at=None)
    structure_logits_processor = None
    sampling_params = SamplingParameters(
        "multinomial",
        1,
        0.9,
        50,
        1.0,
    )
    generator = model.stream(
        prompt, generation_params, structure_logits_processor, sampling_params
    )
    for token in generator:
        assert isinstance(token, str)


def test_exl2_max_chunk_size():
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        cache_q4=True,
        paged=False,
        max_chunk_size=128,
    )
    assert isinstance(model, ExLlamaV2Model)


def test_exl2_cache_default():
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        paged=False,
    )
    assert isinstance(model, ExLlamaV2Model)


def is_flash_attn_available():
    try:
        importlib.import_module("flash_attn")
    except (ImportError, AssertionError):
        return False
    return True


@pytest.mark.skipif(not is_flash_attn_available(), reason="flash-attn is not installed")
def test_exl2_paged():
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        cache_q4=True,
        paged=True,
    )
    assert isinstance(model, ExLlamaV2Model)


def test_exl2_draft_model():
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        draft_model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        cache_q4=True,
        paged=False,
    )
    assert isinstance(model, ExLlamaV2Model)


def test_exl2_draft_model_cache_default():
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        draft_model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        paged=False,
    )
    assert isinstance(model, ExLlamaV2Model)
