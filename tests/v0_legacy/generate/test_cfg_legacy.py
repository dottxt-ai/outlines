import pytest
from dataclasses import asdict

from transformers import AutoModelForCausalLM

from outlines import grammars, models, samplers
from outlines.generator import SteerableGenerator, Generator
from outlines.processors import CFGLogitsProcessor
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.generate.cfg import cfg


@pytest.fixture(scope="session")
def openai_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        return models.openai("gpt-4o")


@pytest.fixture(scope="session")
def llama_cpp_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `llamacpp` function is deprecated",
    ):
        return models.llamacpp(
            "M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            "TinyMistral-248M-v2-Instruct.Q4_K_M.gguf"
        )


@pytest.fixture(scope="session")
def transformers_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers` function is deprecated",
    ):
        return models.transformers("erwanf/gpt2-mini")


@pytest.fixture(scope="session")
def transformers_vision_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `transformers_vision` function is deprecated",
    ):
        return models.transformers_vision(
            "erwanf/gpt2-mini",
            AutoModelForCausalLM
        )


def test_cfg_legacy_openai(openai_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `cfg` function is deprecated",
    ):
        with pytest.raises(
            NotImplementedError,
            match="Cannot use CFG-based structured generation with an OpenAI model"
            + "due to the limitations of the OpenAI API.",
        ):
            cfg(openai_model, grammars.arithmetic)


def test_cfg_legacy_llama_cpp(llama_cpp_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `cfg` function is deprecated",
    ):
        with pytest.raises(
            NotImplementedError,
            match="Not yet available due to bug in llama_cpp tokenizer",
        ):
            cfg(llama_cpp_model, grammars.arithmetic)


def test_cfg_legacy_transformers_vision(transformers_vision_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `cfg` function is deprecated",
    ):
        generator = cfg(transformers_vision_model, grammars.arithmetic)
    assert isinstance(generator, GeneratorVisionV0Adapter)
    assert generator.model == transformers_vision_model
    assert generator.sampling_params == asdict(
        samplers.multinomial().sampling_params
    )
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_vision_model
    assert isinstance(generator.generator.logits_processor, CFGLogitsProcessor)


def test_cfg_legacy_standard_model(transformers_model):
    with pytest.warns(
        DeprecationWarning,
        match="The `cfg` function is deprecated",
    ):
        generator = cfg(transformers_model, grammars.arithmetic, samplers.greedy())
    assert isinstance(generator, GeneratorV0Adapter)
    assert generator.model == transformers_model
    assert generator.sampling_params == asdict(
        samplers.greedy().sampling_params)
    assert isinstance(generator.generator, SteerableGenerator)
    assert generator.generator.model == transformers_model
    assert isinstance(generator.generator.logits_processor, CFGLogitsProcessor)
