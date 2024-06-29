from importlib import reload

import pytest
import torch


def is_metal_available():
    try:
        import mlx.core as mx
        import mlx_lm  # noqa: F401

        assert mx.metal.is_available()
    except (ImportError, AssertionError):
        return False
    return True


def pytest_collection_modifyitems(config, items):
    """
    If mlxlm and Metal aren't available, skip mlxlm tests
    If CUDA isn't available, skip vllm and transformers_vision
    """
    if not torch.cuda.is_available():
        skip_marker = pytest.mark.skip(
            reason="Skipping test because CUDA is not available"
        )
        for item in items:
            if "model_fixture" in item.fixturenames:
                model_param = item.callspec.params.get("model_fixture", None)
                if (
                    model_param.startswith("model_transformers_vision")
                    or model_param.startswith("model_vllm")
                    or model_param.startswith("model_exllamav2")
                ):
                    item.add_marker(skip_marker)

    if not is_metal_available():
        skip_marker = pytest.mark.skip(
            reason="Skipping test because mlx-lm or Metal are not available"
        )
        for item in items:
            if "model_fixture" in item.fixturenames:
                model_param = item.callspec.params.get("model_fixture", None)
                if model_param.startswith("model_mlxlm"):
                    item.add_marker(skip_marker)


@pytest.fixture
def temp_cache_dir():
    import os
    import tempfile

    import outlines.caching
    import outlines.fsm.guide

    with tempfile.TemporaryDirectory() as tempdir:
        os.environ["OUTLINES_CACHE_DIR"] = tempdir
        outlines.caching.get_cache.cache_clear()
        reload(outlines)
        reload(outlines.fsm.guide)
        cache_status = outlines.caching._caching_enabled
        try:
            outlines.caching._caching_enabled = True
            yield
        finally:
            outlines.caching._caching_enabled = cache_status
