import asyncio
import os
import tempfile
import unittest

import diskcache
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
        assert memory.directory == tempdir

        yield outlines.caching.cache()

        memory.clear()


def maybe_await_coroutine(maybe_coro):
    return asyncio.run(maybe_coro) if asyncio.iscoroutine(maybe_coro) else maybe_coro


@pytest.mark.parametrize("use_async", [True, False])
def test_get_cache(test_cache, use_async):
    import outlines

    memory = outlines.get_cache()
    assert isinstance(memory, diskcache.Cache)

    # If the cache is enabled then the size
    # of `store` should not increase the
    # second time `f` is called.
    store = list()

    if use_async:

        @test_cache
        async def f(x):
            store.append(1)
            return x

    else:

        @test_cache
        def f(x):
            store.append(1)
            return x

    maybe_await_coroutine(f(1))
    store_size = len(store)

    maybe_await_coroutine(f(1))
    assert len(store) == store_size

    maybe_await_coroutine(f(2))
    assert len(store) == store_size + 1


@pytest.mark.parametrize("use_async", [True, False])
def test_disable_cache(test_cache, use_async):
    """Make sure that we can disable the cache."""
    import outlines

    outlines.disable_cache()

    # If the cache is disabled then the size
    # of `store` should increase every time
    # `f` is called.
    store = list()

    if use_async:

        @test_cache
        async def f(x):
            store.append(1)
            return x

    else:

        @test_cache
        def f(x):
            store.append(1)
            return x

    maybe_await_coroutine(f(1))
    store_size = len(store)
    maybe_await_coroutine(f(1))
    assert len(store) == store_size + 1


@pytest.mark.parametrize("use_async", [True, False])
def test_clear_cache(test_cache, use_async):
    """Make sure that we can clear the cache."""
    import outlines

    store = list()

    if use_async:

        @test_cache
        async def f(x):
            store.append(1)
            return x

    else:

        @test_cache
        def f(x):
            store.append(1)
            return x

    # The size of `store` does not increase since
    # `f` is cached after the first run.
    maybe_await_coroutine(f(1))
    store_size = len(store)
    maybe_await_coroutine(f(1))
    assert len(store) == store_size

    # The size of `store` should increase if we call `f`
    # after clearing the cache.
    outlines.clear_cache()
    maybe_await_coroutine(f(1))
    assert len(store) == store_size + 1


@pytest.mark.parametrize("use_async", [True, False])
def test_version_upgrade_cache_invalidate(test_cache, use_async, mocker):
    """Ensure we can change the signature of a cached function if we upgrade the version"""

    import outlines.caching

    def simulate_restart_outlines():
        # clearing in-memory lru_cache which returns the diskcache in
        # order to simulate a reload, we're not clearing the diskcache itself
        outlines.caching.get_cache.cache_clear()

    mocker.patch("outlines._version.__version__", new="0.0.0")
    simulate_restart_outlines()

    # initialize cache with signature of Tuple-of-3
    if use_async:

        @test_cache
        async def foo():
            return (1, 2, 3)

    else:

        @test_cache
        def foo():
            return (1, 2, 3)

    a, b, c = maybe_await_coroutine(foo())

    # "restart" outlines without upgrading version
    simulate_restart_outlines()

    # change signature to Tuple-of-2
    if use_async:

        @test_cache
        async def foo():
            return (1, 2)

    else:

        @test_cache
        def foo():
            return (1, 2)

    # assert without version upgrade, old, bad cache is used
    with pytest.raises(ValueError):
        a, b = maybe_await_coroutine(foo())

    # "restart" outlines WITH version upgrade
    mocker.patch("outlines._version.__version__", new="0.0.1")
    simulate_restart_outlines()

    # change signature to Tuple-of-2
    if use_async:

        @test_cache
        async def foo():
            return (1, 2)

    else:

        @test_cache
        def foo():
            return (1, 2)

    # assert with version upgrade, old cache is invalidated and new cache is used
    a, b = maybe_await_coroutine(foo())


@pytest.mark.parametrize("use_async", [True, False])
def test_cache_disabled_decorator(test_cache, use_async):
    """Ensure cache can be disabled in a local scope"""

    from outlines.caching import cache_disabled

    mock = unittest.mock.MagicMock()

    if use_async:

        @test_cache
        async def fn():
            mock()
            return 1

    else:

        @test_cache
        def fn():
            mock()
            return 1

    # first call isn't cached
    maybe_await_coroutine(fn())
    assert mock.call_count == 1

    # second call doesn't run fn, uses cache
    maybe_await_coroutine(fn())
    assert mock.call_count == 1

    # cache_disabled decorator disables cache within scope
    with cache_disabled():
        maybe_await_coroutine(fn())
    assert mock.call_count == 2  # called once in cache_disabled scope

    # scope has exited, cache is enabled again
    maybe_await_coroutine(fn())
    assert mock.call_count == 2
