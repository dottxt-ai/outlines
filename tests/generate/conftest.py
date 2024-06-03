from importlib import reload

import pytest


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
