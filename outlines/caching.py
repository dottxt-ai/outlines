import hashlib
import os
from typing import Callable

import cloudpickle
import torch
from diskcache import Cache

home_dir = os.path.expanduser("~")
cache_dir = os.environ.get("OUTLINES_CACHE_DIR", f"{home_dir}/.cache/outlines")
memory = Cache(cache_dir, eviction_policy="none", cull_limit=0)
_caching_enabled = True


def hash_data(*data) -> str:
    """Pickles and hashes all the data passed to it as args."""
    result = hashlib.md5()  # nosec B303
    for datum in data:
        if isinstance(datum, torch.Tensor):
            datum = datum.cpu().numpy()
        result.update(cloudpickle.dumps(datum))
    return result.hexdigest()


def cache(key_function: Callable):
    """Caching decorator for memoizing function calls based on a provided key.

    Parameters
    ----------
    key
      A callable function used to generate a unique key for each function call.

    Returns
    -------
      A decorator function that can be applied to other functions.
    """

    def decorator(cached_function: Callable):
        def wrapper(*args, **kwargs):
            if not _caching_enabled:
                return cached_function(*args, **kwargs)
            key_args = key_function(*args, **kwargs)
            cache_key = hash_data(*key_args)
            if cache_key in memory:
                return memory[cache_key]
            result = cached_function(*args, **kwargs)
            memory[cache_key] = result
            return result

        return wrapper

    return decorator


def get_cache():
    """Get the context object that contains previously-computed return values.

    The cache is used to avoid unnecessary computations and API calls, which can
    be long and expensive for large models.

    The cache directory defaults to `HOMEDIR/.cache/outlines`, but this choice
    can be overridden by the user by setting the value of the `OUTLINES_CACHE_DIR`
    environment variable.

    """
    return memory


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
    global memory
    memory.clear()
