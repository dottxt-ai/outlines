import hashlib
import pickle
import sys


def hash_dict(d) -> str:
    def make_hashable(obj):
        if isinstance(obj, (bool, int, float, str, type(None))):
            if isinstance(obj, str):
                return sys.intern(obj)
            return obj
        if isinstance(obj, dict):
            return tuple(sorted(
                (sys.intern(k) if isinstance(k, str) else k, make_hashable(v))
                for k, v in obj.items()
            ))
        if isinstance(obj, (list, tuple)):
            return tuple(make_hashable(e) for e in obj)
        return str(obj)

    hashable_obj = make_hashable(d)
    pickled_obj = pickle.dumps(hashable_obj, protocol=4)
    return hashlib.sha256(pickled_obj).hexdigest()
