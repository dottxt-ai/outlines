import importlib
from unittest.mock import patch

import pytest

import outlines.models as models
from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.models.exllamav2 import ExLlamaV2Model


@pytest.fixture(scope="session")
def model_exllamav2(tmp_path_factory):
    return models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        cache_q4=True,
        paged=False,
    )


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_exl2_import_error(request, model_fixture):
    with patch.dict("sys.modules", {"exllamav2": None}):
        with pytest.raises(ImportError):
            models.exl2(
                model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
                cache_q4=True,
                paged=False,
            )


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_attributes(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    assert hasattr(model, "generator")
    assert hasattr(model, "tokenizer")
    assert model.tokenizer.convert_token_to_string(1) == 1
    assert hasattr(model, "max_seq_len")
    assert isinstance(model.max_seq_len, int)


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_generate_prompt_types(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_generate_no_max_tokens(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_generate_test_stop_at(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_generate_multisampling(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_prepare_generation_parameters(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_stream_prompt_types(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_stream_no_max_tokens(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_stream_test_stop_at(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_stream_multisampling(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_model_stream_seed(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
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


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_reformat_output(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    sampling_params = SamplingParameters(
        "multinomial",
        1,
    )
    output = "test"
    reformatted_output = model.reformat_output(output, sampling_params)
    assert reformatted_output == output
    output = ["test"]
    reformatted_output = model.reformat_output(output, sampling_params)
    assert reformatted_output == output[0]
    output = ["test", "test"]
    sampling_params = SamplingParameters(
        "multinomial",
        1,
    )
    reformatted_output = model.reformat_output(output, sampling_params)
    assert len(reformatted_output) == 2
    assert reformatted_output[0] == "test"
    assert reformatted_output[1] == "test"
    output = ["test", "test"]
    sampling_params = SamplingParameters(
        "multinomial",
        2,
    )
    reformatted_output = model.reformat_output(output, sampling_params)
    assert len(reformatted_output) == 2
    assert reformatted_output[0] == "test"
    assert reformatted_output[1] == "test"
    output = ["test", "test", "test", "test"]
    sampling_params = SamplingParameters(
        "multinomial",
        2,
    )
    reformatted_output = model.reformat_output(output, sampling_params)
    assert len(reformatted_output) == 2
    assert reformatted_output[0] == ["test", "test"]
    assert reformatted_output[1] == ["test", "test"]


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_exl2_max_chunk_size(request, model_fixture):
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        cache_q4=True,
        paged=False,
        max_chunk_size=128,
    )
    assert isinstance(model, ExLlamaV2Model)


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_exl2_cache_default(request, model_fixture):
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        paged=False,
    )
    assert isinstance(model, ExLlamaV2Model)


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def is_flash_attn_available():
    try:
        importlib.import_module("flash_attn")
    except (ImportError, AssertionError):
        return False
    return True


@pytest.mark.skipif(not is_flash_attn_available(), reason="flash-attn is not installed")
@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_exl2_paged(request, model_fixture):
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        cache_q4=True,
        paged=True,
    )
    assert isinstance(model, ExLlamaV2Model)


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_exl2_draft_model(request, model_fixture):
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        draft_model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        cache_q4=True,
        paged=False,
    )
    assert isinstance(model, ExLlamaV2Model)


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_exl2_draft_model_cache_default(request, model_fixture):
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        draft_model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        paged=False,
    )
    assert isinstance(model, ExLlamaV2Model)


@pytest.mark.parametrize("model_fixture", ["model_exllamav2"])
def test_exl2_set_max_seq_len(request, model_fixture):
    model = models.exl2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        max_seq_len=2048,
        paged=False,
        cache_q4=True,
    )
    assert isinstance(model, ExLlamaV2Model)
