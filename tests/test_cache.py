import os
import tempfile
from pathlib import Path

import perscache
import pytest


@pytest.fixture
def refresh_environment():
    """Refresh the test environment.

    This deletes any reference to `outlines` in the modules dictionary and unsets the
    `OUTLINES_CACHE_DIR` environment variable if set. This is necessary because we
    are using a module variable to hold the cache.

    """
    import sys

    for key in list(sys.modules.keys()):
        if "outlines" in key:
            del sys.modules[key]

    try:
        del os.environ["OUTLINES_CACHE_DIR"]
    except KeyError:
        pass


@pytest.fixture
def test_cache(refresh_environment):
    """Initialize a temporary cache and delete it after the test has run."""
    with tempfile.TemporaryDirectory() as tempdir:
        os.environ["OUTLINES_CACHE_DIR"] = tempdir
        import outlines

        memory = outlines.get_cache()
        assert memory.storage.location == Path(tempdir)

        yield outlines.caching.cache()

        memory.storage.clear()


def test_get_cache(test_cache):
    import outlines

    memory = outlines.get_cache()
    assert isinstance(memory, perscache.Cache)
    assert isinstance(memory.storage, perscache.storage.LocalFileStorage)

    # If the cache is enabled then the size
    # of `store` should not increase the
    # second time `f` is called.
    store = list()

    @test_cache
    def f(x):
        store.append(1)
        return x

    f(1)
    store_size = len(store)

    f(1)
    assert len(store) == store_size

    f(2)
    assert len(store) == store_size + 1


def test_disable_cache(test_cache):
    """Make sure that we can disable the cache."""
    import outlines

    outlines.disable_cache()

    # If the cache is disabled then the size
    # of `store` should increase every time
    # `f` is called.
    store = list()

    @test_cache
    def f(x):
        store.append(1)
        return x

    f(1)
    store_size = len(store)
    f(1)
    assert len(store) == store_size + 1


def test_clear_cache(test_cache):
    """Make sure that we can clear the cache."""
    import outlines

    store = list()

    @test_cache
    def f(x):
        store.append(1)
        return x

    # The size of `store` does not increase since
    # `f` is cached after the first run.
    f(1)
    store_size = len(store)
    f(1)
    assert len(store) == store_size

    # The size of `store` should increase if we call `f`
    # after clearing the cache.
    outlines.clear_cache()
    f(1)
    assert len(store) == store_size + 1
