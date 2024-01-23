import importlib

import interegular
import numba

import outlines

outlines.disable_cache()


def test_benchmark_compile_numba(benchmark, tokenizer, mocker):
    """Compile a basic regex to benchmark the numba compilation time"""

    def setup():
        from outlines.fsm import regex

        original_njit = numba.njit

        def mock_njit(*args, **kwargs):
            kwargs["cache"] = False
            return original_njit(*args, **kwargs)

        mocker.patch("numba.njit", new=mock_njit)
        importlib.reload(regex)

        regex_pattern, _ = regex.make_deterministic_fsm(
            interegular.parse_pattern("a").to_fsm().reduce()
        )
        return (regex, regex_pattern, tokenizer), {}

    benchmark.pedantic(
        lambda r, *args: r.create_fsm_index_tokenizer(*args), rounds=2, setup=setup
    )
