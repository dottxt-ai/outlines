import asyncio

import pytest

from outlines import elemwise


def test_single_input_sync():
    @elemwise
    def sync_function(query):
        return query

    result = sync_function("test")
    assert result == "test"

    result = sync_function(["test1", "test2"])
    assert result == ["test1", "test2"]


def test_single_input_async():
    @elemwise
    async def async_function(query):
        return query

    result = asyncio.run(async_function("test"))
    assert result == "test"

    result = asyncio.run(async_function(["test1", "test2"]))
    assert result == ["test1", "test2"]


def test_wrong_dimensions():
    @elemwise
    def sync_function(a, b):
        return a + b

    with pytest.raises(TypeError, match="All lists"):
        sync_function(["a"], ["b", "c"])

    @elemwise
    async def async_function(a, b):
        return a + b

    with pytest.raises(TypeError, match="All lists"):
        async_function(["a"], ["b", "c"])


def test_lists():
    @elemwise
    def sync_function(a, b):
        return a + b

    result = sync_function(["a", "b"], ["c", "d"])
    assert result == ["ac", "bd"]

    @elemwise
    async def async_function(a, b):
        return a + b

    result = asyncio.run(async_function(["a", "b"], ["c", "d"]))
    assert result == ["ac", "bd"]


def test_broadcasting():
    @elemwise
    def sync_function(a, b):
        return a + b

    result = sync_function(["a", "b"], "c")
    assert result == ["ac", "bc"]

    @elemwise
    async def async_function(a, b):
        return a + b

    result = asyncio.run(async_function(["a", "b"], "c"))
    assert result == ["ac", "bc"]


def test_kwargs():
    @elemwise
    def sync_function(a, b="b"):
        return a + b

    result = sync_function(["a", "c"])
    assert result == ["ab", "cb"]

    result = sync_function(["a", "c"], b="d")
    assert result == ["ad", "cd"]

    @elemwise
    async def async_function(a, b="b"):
        return a + b

    result = asyncio.run(async_function(["a", "c"]))
    assert result == ["ab", "cb"]

    result = asyncio.run(async_function(["a", "c"], b="d"))
    assert result == ["ad", "cd"]
