import importlib
import os

import pytest

ENABLE_CACHE_TEST_PREFIXES = ["tests.test_cache", "tests.benchmark"]


@pytest.fixture(scope="module", autouse=True)
def disable_cache(request):
    for prefix in ENABLE_CACHE_TEST_PREFIXES:
        if request.module.__name__.startswith(prefix):
            return

    os.environ["OUTLINES_DISABLE_CACHE"] = "1"
    import outlines.caching

    importlib.reload(outlines.caching)
