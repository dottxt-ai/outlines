import asyncio
import os
import tempfile
import threading
import unittest
from importlib import reload

import diskcache
import pytest
from diskcache import Cache, UNKNOWN
from outlines.caching import CloudpickleDisk


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    directory = tempfile.mkdtemp()
    yield directory


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
        assert memory.directory == tempdir

        yield outlines.caching.cache()

        memory.clear()


def test_get_cache(test_cache):
    import outlines

    memory = outlines.get_cache()
    assert isinstance(memory, diskcache.Cache)

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


def test_version_upgrade_cache_invalidate(test_cache, mocker):
    """Ensure we can change the signature of a cached function if we upgrade the version"""

    import outlines.caching

    def simulate_restart_outlines():
        # clearing in-memory lru_cache which returns the diskcache in
        # order to simulate a reload, we're not clearing the diskcache itself
        outlines.caching.get_cache.cache_clear()

    mocker.patch("outlines._version.__version__", new="0.0.0")
    simulate_restart_outlines()

    # initialize cache with signature of Tuple-of-3
    @test_cache
    def foo():
        return (1, 2, 3)

    a, b, c = foo()

    # "restart" outlines without upgrading version
    simulate_restart_outlines()

    # change signature to Tuple-of-2
    @test_cache
    def foo():
        return (1, 2)

    # assert without version upgrade, old, bad cache is used
    with pytest.raises(ValueError):
        a, b = foo()

    # "restart" outlines WITH version upgrade
    mocker.patch("outlines._version.__version__", new="0.0.1")
    simulate_restart_outlines()

    # change signature to Tuple-of-2
    @test_cache
    def foo():
        return (1, 2)

    # assert with version upgrade, old cache is invalidated and new cache is used
    a, b = foo()


def test_cache_disabled_decorator(test_cache):
    """Ensure cache can be disabled in a local scope"""

    from outlines.caching import cache_disabled

    mock = unittest.mock.MagicMock()

    @test_cache
    def fn():
        mock()
        return 1

    # first call isn't cached
    fn()
    assert mock.call_count == 1

    # second call doesn't run fn, uses cache
    fn()
    assert mock.call_count == 1

    # cache_disabled decorator disables cache within scope
    with cache_disabled():
        fn()
    assert mock.call_count == 2  # called once in cache_disabled scope

    # scope has exited, cache is enabled again
    fn()
    assert mock.call_count == 2


@pytest.mark.asyncio
async def test_cache_disabled_does_not_leak_to_concurrent_tasks():
    """Ensure cache_disabled does not affect tasks that did not enter it"""

    from outlines.caching import _is_caching_enabled, cache_disabled

    disabled = asyncio.Event()
    observed = asyncio.Event()
    states = {}

    async def with_cache_disabled():
        with cache_disabled():
            disabled.set()
            await observed.wait()
            states["inside"] = _is_caching_enabled()

    async def concurrent():
        await disabled.wait()
        states["outside"] = _is_caching_enabled()
        observed.set()

    await asyncio.gather(with_cache_disabled(), concurrent())

    assert states["inside"] is False
    assert states["outside"] is True


@pytest.mark.asyncio
async def test_cache_disabled_restores_state_across_await():
    """Ensure cache_disabled restores the previous state when the block awaits"""

    from outlines.caching import _is_caching_enabled, cache_disabled

    with cache_disabled():
        await asyncio.sleep(0)
        assert _is_caching_enabled() is False

    assert _is_caching_enabled() is True


def test_disable_cache_applies_to_other_threads():
    """Ensure disable_cache remains a process-wide switch"""

    import outlines.caching

    states = []
    original_state = outlines.caching._caching_enabled
    try:
        outlines.caching.disable_cache()
        thread = threading.Thread(
            target=lambda: states.append(outlines.caching._is_caching_enabled())
        )
        thread.start()
        thread.join()
    finally:
        outlines.caching._caching_enabled = original_state

    assert states == [False]


@pytest.fixture
def temp_cache_dir():
    import os
    import tempfile

    import outlines.caching

    with tempfile.TemporaryDirectory() as tempdir:
        os.environ["OUTLINES_CACHE_DIR"] = tempdir
        outlines.caching.get_cache.cache_clear()
        reload(outlines)
        cache_status = outlines.caching._caching_enabled
        try:
            outlines.caching._caching_enabled = True
            yield
        finally:
            outlines.caching._caching_enabled = cache_status
