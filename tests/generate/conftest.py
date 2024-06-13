from importlib import reload

import pytest


def pytest_collection_modifyitems(config, items):
    """If mlxlm and Metal aren't available, skip mlxlm tests"""
    try:
        import mlx.core as mx
        import mlx_lm  # noqa: F401

        assert mx.metal.is_available()
    except (ImportError, AssertionError):
        skip_marker = pytest.mark.skip(
            reason="Skipping test because mlx-lm or Metal are not available"
        )
        for item in items:
            if "model_fixture" in item.fixturenames:
                model_param = item.callspec.params.get("model_fixture", None)
                if model_param == "model_mlxlm":
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
