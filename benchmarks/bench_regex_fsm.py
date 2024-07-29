import random

from outlines.caching import cache_disabled
from outlines.fsm.regex import reduced_vocabulary
from outlines.models.tokenizer import Tokenizer

from .common import ensure_numba_compiled


class MockTokenizer(Tokenizer):
    def __init__(self, token_strs):
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.special_tokens = {0, 1}

        self.vocabulary = {"<eos>": 0, "<pad>": 1}

        for i, tok in enumerate(token_strs):
            self.vocabulary[tok] = i + 2

    @classmethod
    def from_random_tokens(cls, n_tokens, max_token_length=8, seed=42):
        random.seed(seed)
        tokens = [
            "".join(
                chr(random.randint(0, 4096))
                for __ in range(random.randint(0, max_token_length))
            )
            for _ in range(n_tokens)
        ]
        return cls(tokens)

    def convert_token_to_string(self, token):
        return token

    def __hash__(self):
        return hash(tuple(sorted(self.vocabulary.items())))


def reduced_vocabulary_uncached(*args, **kwargs):
    return reduced_vocabulary.__wrapped__(*args, **kwargs)


class RegexReducedVocabularyBenchmark:
    params = [10000, 100000, 1000000]
    param_names = ["vocab_size"]

    def setup(self, vocab_size):
        ensure_numba_compiled(MockTokenizer([chr(i) for i in range(128)]))

        self.tokenizer = MockTokenizer.from_random_tokens(vocab_size)

    @cache_disabled()
    def time_reduced_vocabulary(self, _):
        reduced_vocabulary_uncached(self.tokenizer)
