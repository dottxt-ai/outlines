# TODO: Add diskcache in dependencies
from diskcache import Cache
import torch
import time

cache = Cache('/home/avirlrma/.cache/diskcache')

def cache_function(cache):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Convert tensor arguments to bytes for caching
            cache_key = tuple(arg.numpy().tobytes() if 
                              torch.is_tensor(arg) else arg for arg in args)
            if cache_key in cache:
                return cache[cache_key]
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return result
        return wrapper
    return decorator

token_ids = torch.full((2, 3), 6)
attention_mask = torch.full((2, 3), 5)
kv_cache = torch.full((2, 3), 3)

def cache_key(tensor1, tensor2, tensor3):
    # Convert tensors to bytes and concatenate to form a unique key
    return (tensor1.numpy().tobytes(), tensor2.numpy().tobytes(), tensor3.numpy().tobytes())

@cache_function(cache)
def func(token_ids, attention_mask, kv_cache):
    # mimiciking the expensive operation
    time.sleep(4)
    return "hello"


if __name__ == '__main__':
    print(func(token_ids, attention_mask, kv_cache))
    # key = cache_key(token_ids, attention_mask, kv_cache)
    # print(len(cache))
    # if key in cache:
    #     print("key in cache")
    #     print(cache[key])
    # else:
    #     cache[key] = func(token_ids, attention_mask, kv_cache)
    #     print(cache[key])
    cache.close()
