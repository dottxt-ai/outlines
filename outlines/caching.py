import os
from typing import Callable, Optional
import hashlib
import cloudpickle
import torch
from perscache import Cache, NoCache
from perscache.serializers import JSONSerializer
from perscache.storage import LocalFileStorage
from diskcache import Cache as DiskCache

home_dir = os.path.expanduser("~")
cache_dir = os.environ.get("OUTLINES_CACHE_DIR", f"{home_dir}/.cache/outlines")
memory = Cache(serializer=JSONSerializer(), storage=LocalFileStorage(cache_dir))
diskcache_memory = DiskCache(cache_dir, eviction_policy = 'none', cull_limit=1000)

def cache(ignore: Optional[str] = None):
    def cache_fn(fn: Callable):
        return memory.cache(ignore=ignore)(fn)

    return cache_fn


def hash_it(*data) -> str:
    """Pickles and hashes all the data passed to it as args."""
    result = hashlib.md5()  # nosec B303
    for datum in data:
        if isinstance(datum, torch.Tensor):
            datum = datum.cpu().numpy()
        result.update(cloudpickle.dumps(datum))
    return result.hexdigest()

def diskcache(cache_key_args_func):
    def decorator(func):
        def wrapper(*args, **kwargs):
            key_args, key_kwargs = cache_key_args_func(*args, **kwargs)
            cache_key = hash_it(*key_args, **key_kwargs)
            if cache_key in diskcache_memory:
                return diskcache_memory[cache_key]
            result = func(*args, **kwargs)
            diskcache_memory[cache_key] = result
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
    global memory
    memory = NoCache()


def clear_cache():
    """Erase the cache completely."""
    global memory
    memory.storage.clear()
