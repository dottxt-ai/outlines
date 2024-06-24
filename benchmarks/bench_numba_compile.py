import importlib

import interegular
import numba

from outlines.caching import cache_disabled
from outlines.fsm import regex

from .common import setup_tokenizer


class NumbaCompileBenchmark:
    def setup(self):
        self.tokenizer = setup_tokenizer()
        self.regex = regex
        original_njit = numba.njit

        def mock_njit(*args, **kwargs):
            kwargs["cache"] = False
            return original_njit(*args, **kwargs)

        self.original_njit = original_njit
        numba.njit = mock_njit
        importlib.reload(self.regex)
        self.regex_pattern, _ = self.regex.make_deterministic_fsm(
            interegular.parse_pattern("a").to_fsm().reduce()
        )

    def teardown(self):
        numba.njit = self.original_njit

    @cache_disabled()
    def time_compile_numba(self):
        self.regex.create_fsm_index_tokenizer(self.regex_pattern, self.tokenizer)
