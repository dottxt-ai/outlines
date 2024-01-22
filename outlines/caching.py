import asyncio
import functools
import hashlib
import os
from typing import Callable, Optional

import cloudpickle
from diskcache import Cache

_caching_enabled = True


@functools.lru_cache(1)
def get_cache():
    """Get the context object that contains previously-computed return values.

    The cache is used to avoid unnecessary computations and API calls, which can
    be long and expensive for large models.

    The cache directory defaults to `HOMEDIR/.cache/outlines`, but this choice
    can be overridden by the user by setting the value of the `OUTLINES_CACHE_DIR`
    environment variable.

    """
    from outlines._version import __version__ as outlines_version  # type: ignore

    home_dir = os.path.expanduser("~")
    cache_dir = os.environ.get("OUTLINES_CACHE_DIR", f"{home_dir}/.cache/outlines")
    memory = Cache(cache_dir, eviction_policy="none", cull_limit=0)

    # ensure if version upgrade occurs, old cache is pruned
    if outlines_version != memory.get("__version__"):
        memory.clear()
    memory["__version__"] = outlines_version

    return memory


def hash_arguments(*args, **kwargs) -> str:
    """Create a hash out of the args and kwargs provided"""
    result = hashlib.md5()
    for item in list(args) + sorted(kwargs.items()):
        result.update(cloudpickle.dumps(item))
    return result.hexdigest()


def cache(key_function: Optional[Callable] = None):
    """Caching decorator for memoizing function calls.
    The cache key is created based on the values returned by the key_function callable
    if provided or based on the arguments of the decorated function directly otherwise
    Parameters
    ----------
    key_function
      A callable function used to generate a unique key for each function call. It's
      called with the arguments of the decorated function as arguments
    Returns
    -------
      A decorator function that can be applied to other functions.
    """

    def decorator(cached_function: Callable):
        memory = get_cache()

        def wrapper(*args, **kwargs):
            if not _caching_enabled:
                return cached_function(*args, **kwargs)
            if key_function:
                key_args = key_function(*args, **kwargs)
                cache_key = hash_arguments(*key_args)
            else:
                cache_key = hash_arguments(*args, **kwargs)
            if cache_key in memory:
                return memory[cache_key]
            result = cached_function(*args, **kwargs)
            memory[cache_key] = result
            return result

        async def async_wrapper(*args, **kwargs):
            if not _caching_enabled:
                return await cached_function(*args, **kwargs)
            if key_function:
                key_args = key_function(*args, **kwargs)
                cache_key = hash_arguments(*key_args)
            else:
                cache_key = hash_arguments(*args, **kwargs)
            if cache_key in memory:
                return memory[cache_key]
            result = await cached_function(*args, **kwargs)
            memory[cache_key] = result
            return result

        if asyncio.iscoroutinefunction(cached_function):
            return async_wrapper
        else:
            return wrapper

    return decorator


def disable_cache():
    """Disable the cache for this session.

    Generative models output different results each time they are called when
    sampling. This can be a desirable property for some workflows, in which case
    one can call `outlines.call.disable` to disable the cache for the session.

    This function does not delete the cache, call `outlines.cache.clear`
    instead. It also does not overwrite the cache with the values returned
    during the session.

    Example
    -------

    `outlines.cache.disable` should be called right after importing outlines:

    >>> import outlines.cache as cache
    >>> cache.disable()

    """
    global _caching_enabled
    _caching_enabled = False


def clear_cache():
    """Erase the cache completely."""
    memory = get_cache()
    memory.clear()
