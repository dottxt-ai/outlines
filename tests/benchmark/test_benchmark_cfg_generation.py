from pathlib import Path

import pytest
from conftest import add_test_output  # type: ignore

import outlines
import outlines.grammars
from outlines.fsm.fsm import CFGFSM, FSMState
from outlines.fsm.regex import reduced_vocabulary

# Collects all samples within cfg_samples/ and makes adding
# a test case as easy as adding a valid sample to cfg_samples/
all_samples = {}
examples_path = Path(__file__).parent / "cfg_samples"
for sample_collection_path in examples_path.iterdir():
    grammar_name = sample_collection_path.name
    grammar = getattr(outlines.grammars, grammar_name)
    for sample_path in sample_collection_path.iterdir():
        test_name = f"{grammar_name}_{sample_path.name}"
        with open(sample_path) as f:
            all_samples[test_name] = (grammar, f.read().rstrip("\n"))


class MockGenerator:
    def __init__(self, cfg_fsm, to_generate):
        self.cfg_fsm = cfg_fsm
        self.to_generate = to_generate

        self._generated_token_ids = []

        # precompute legal tokens at each step rather than on the fly to
        # ensure we're only measuring the performance of the logits processor
        self.accepted_token_ids_map = {}
        self.reverse_vocab = {}

        vocab, _ = reduced_vocabulary(cfg_fsm.tokenizer)
        vocab = dict(vocab)
        max_token_len = max(map(len, vocab))
        for i in range(len(to_generate)):
            accepted_token_ids = set()
            remaining_generate = to_generate[i:]
            for j in range(max_token_len):
                tok = remaining_generate[: j + 1]
                if tok in vocab:
                    new_accepted = set(vocab[tok])
                    for tid in new_accepted:
                        self.reverse_vocab[tid] = tok
                    accepted_token_ids |= new_accepted
            self.accepted_token_ids_map[remaining_generate] = accepted_token_ids

    def run_until_eos(self):
        num_tokens_generated = 0
        generated = ""

        state = FSMState(0)
        while self.to_generate:
            num_tokens_generated += 1

            allowed_token_ids = set(self.cfg_fsm.allowed_token_ids(state))
            expected_token_ids = self.accepted_token_ids_map[self.to_generate]

            # TODO: Create an issue. Currently not all generations are legal,
            # for example the tokenizer has the token `{"` however because { and " are
            # separate terminals, thus {" isn't considered a legal token

            # After https://github.com/outlines-dev/outlines/issues/573 resolution,
            # this should simply assert that expected_token_ids.issubset(allowed_token_ids)
            candidate_token_ids = allowed_token_ids & expected_token_ids
            if not candidate_token_ids:
                tok2str = lambda token_id: self.cfg_fsm.tokenizer.decode([token_id])[0]
                raise Exception(
                    "\n".join(
                        [
                            "Failed to produce any tokens which would complete the sampled grammar",
                            f"expected_tokens: {list(map(tok2str, expected_token_ids))}",
                            f"allowed_tokens: {list(map(tok2str, allowed_token_ids))}",
                            f"lexed output: {list(self.cfg_fsm.parser.lex(generated))}",
                            f"raw output: '{generated}'",
                        ]
                    )
                )

            next_token_id = sorted(candidate_token_ids)[
                0
            ]  # make deterministic for cache testing
            next_token_str = self.reverse_vocab[next_token_id]

            assert self.to_generate.startswith(next_token_str)
            self.to_generate = self.to_generate[len(next_token_str) :]
            generated += next_token_str

            state = self.cfg_fsm.next_state(state=state, token_id=next_token_id)

        assert self.cfg_fsm.tokenizer.eos_token_id in set(
            self.cfg_fsm.allowed_token_ids(state)
        ), "Failed to produce EOS token at end of generation"

        return num_tokens_generated


@pytest.mark.benchmark_cfg
@pytest.mark.parametrize("preload_cache", [True, False])
@pytest.mark.parametrize("sample_name", all_samples.keys())
def test_benchmark_cfg_generation(
    request, benchmark, tokenizer, ensure_numba_compiled, sample_name, preload_cache
):
    """Benchmark CFGLogitsProcessor Generation"""

    def get_mock_generator():
        # don't let residual cache impact results
        outlines.clear_cache()

        cfg, sample = all_samples[sample_name]
        cfg_fsm = CFGFSM(cfg, tokenizer)

        if preload_cache:
            # precompute the RegexFSM cache by running once
            MockGenerator(
                cfg_fsm=cfg_fsm.copy(),
                to_generate=sample,
            ).run_until_eos()

        mock_gen = MockGenerator(
            cfg_fsm=cfg_fsm,
            to_generate=sample,
        )
        return (mock_gen,), {}

    num_tokens = benchmark.pedantic(
        lambda mock_gen: mock_gen.run_until_eos(),
        setup=get_mock_generator,
    )

    output_str = "\n".join(
        [
            "{}:",
            "\tTokens / Second: {:.3f}",
            "\t(Num Tokens: {}, Time: {:.3f} seconds)",
        ]
    ).format(
        request.node.nodeid,
        num_tokens / benchmark.stats.stats.mean,
        num_tokens,
        benchmark.stats.stats.mean,
    )
    add_test_output(output_str)
