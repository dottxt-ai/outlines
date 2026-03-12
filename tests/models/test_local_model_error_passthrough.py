import sys
import types

import pytest

from outlines.models.mlxlm import MLXLM
from outlines.models.transformers import Transformers
from outlines.models.vllm_offline import VLLMOffline


def test_transformers_passthrough_native_generation_errors():
    class DummyModel:
        class config:
            is_encoder_decoder = False

        def generate(self, **kwargs):
            raise RuntimeError("local generation failed")

    model = Transformers.__new__(Transformers)
    model.model = DummyModel()

    with pytest.raises(RuntimeError, match="local generation failed"):
        model._generate_output_seq("", {"input_ids": object()})


def test_mlxlm_passthrough_native_generation_errors(monkeypatch):
    def _raise_generate(*args, **kwargs):
        raise ValueError("mlx local failure")

    fake_mlx_lm = types.SimpleNamespace(generate=_raise_generate)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    model = MLXLM.__new__(MLXLM)
    model.model = object()
    model.mlx_tokenizer = object()
    model.type_adapter = types.SimpleNamespace(
        format_input=lambda _: "prompt",
        format_output_type=lambda _: None,
    )

    with pytest.raises(ValueError, match="mlx local failure"):
        model.generate("hello")


def test_vllm_offline_passthrough_native_generation_errors():
    class DummyModel:
        def generate(self, **kwargs):
            raise RuntimeError("vllm offline failed")

    model = VLLMOffline.__new__(VLLMOffline)
    model.model = DummyModel()
    model.type_adapter = types.SimpleNamespace(format_input=lambda _: "prompt")
    model._build_generation_args = lambda _inference_kwargs, _output_type: object()

    with pytest.raises(RuntimeError, match="vllm offline failed"):
        model.generate("hello")
