"""Sequence state must follow beam ancestry, not batch row positions."""

import re

import numpy as np
import pytest
import torch
from outlines_core import Index, Vocabulary

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor

TOKENS = {char: i + 1 for i, char in enumerate("abcdxyzwqrs")}
VOCABULARY = Vocabulary(0, {char: [idx] for char, idx in TOKENS.items()})
VOCAB_SIZE = len(TOKENS) + 1


def allowed(processor, prefixes, library, prompts=None):
    if prompts is None:
        prompts = [[99, 99] for _ in prefixes]
    ids = [
        prompt + [TOKENS[c] for c in text]
        for prompt, text in zip(prompts, prefixes)
    ]
    if library == "torch":
        logits = processor(torch.tensor(ids), torch.zeros(len(ids), VOCAB_SIZE))
        return [
            set(torch.isfinite(row).nonzero().flatten().tolist()) for row in logits
        ]
    logits = processor(
        np.array(ids), np.zeros((len(ids), VOCAB_SIZE), dtype=np.float32)
    )
    return [set(np.flatnonzero(np.isfinite(row))) for row in logits]


@pytest.mark.parametrize("library", ["torch", "numpy"])
@pytest.mark.parametrize(
    "regex,second",
    [("acx|bdy", ["bd", "ac"]), ("acx|bcy", ["bc", "ac"])],
)
def test_reordered_beams(library, regex, second):
    processor = OutlinesCoreLogitsProcessor(Index(regex, VOCABULARY), library)
    allowed(processor, ["", ""], library)
    allowed(processor, ["a", "b"], library)
    assert allowed(processor, second, library) == [{TOKENS["y"]}, {TOKENS["x"]}]


@pytest.mark.parametrize("library", ["torch", "numpy"])
def test_forked_beams_have_independent_states(library):
    processor = OutlinesCoreLogitsProcessor(Index("ac(xq|zr)|bdy", VOCABULARY), library)
    allowed(processor, ["", ""], library)
    allowed(processor, ["a", "b"], library)
    assert allowed(processor, ["ac", "ac"], library) == [
        {TOKENS["x"], TOKENS["z"]}
    ] * 2
    assert allowed(processor, ["acx", "acz"], library) == [
        {TOKENS["q"]}, {TOKENS["r"]}
    ]
    assert allowed(processor, ["aczr", "acxq"], library) == [{0}, {0}]


@pytest.mark.parametrize("library", ["torch", "numpy"])
def test_fork_after_rollback_history_is_exhausted(library):
    processor = OutlinesCoreLogitsProcessor(
        Index("a{40}(xq|zr)|b{40}dy", VOCABULARY), library
    )
    allowed(processor, ["", ""], library)
    for length in range(1, 41):
        allowed(processor, ["a" * length, "b" * length], library)
    assert allowed(processor, ["a" * 40 + "x", "a" * 40 + "z"], library) == [
        {TOKENS["q"]}, {TOKENS["r"]}
    ]
    assert allowed(processor, ["a" * 40 + "zr", "a" * 40 + "xq"], library) == [
        {0}, {0}
    ]


@pytest.mark.parametrize("library", ["torch", "numpy"])
def test_reuse_finished_beam_with_ignored_padding(library):
    processor = OutlinesCoreLogitsProcessor(Index("a|bc(xq|zr)", VOCABULARY), library)
    allowed(processor, ["", ""], library)
    allowed(processor, ["a", "b"], library)
    # A finished row can receive padding that does not advance its guide.
    assert allowed(processor, ["aq", "bc"], library) == [
        {0}, {TOKENS["x"], TOKENS["z"]}
    ]
    assert allowed(processor, ["bcx", "bcz"], library) == [
        {TOKENS["q"]}, {TOKENS["r"]}
    ]
    assert allowed(processor, ["bczr", "bcxq"], library) == [{0}, {0}]


@pytest.mark.parametrize("library", ["torch", "numpy"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_stable_batch_padding_and_reset(library, batch_size):
    processor = OutlinesCoreLogitsProcessor(Index("acx|bcy", VOCABULARY), library)
    for prompts in ([[0, 0, 99], [0, 98, 99]], [[97], [96]]):
        prompts = prompts[:batch_size]
        processor.reset()
        assert allowed(processor, [""] * batch_size, library, prompts) == [
            {TOKENS["a"], TOKENS["b"]}
        ] * batch_size
        for prefix, next_token in [("a", "c"), ("ac", "x")]:
            assert allowed(processor, [prefix] * batch_size, library, prompts) == [
                {TOKENS[next_token]}
            ] * batch_size
        assert allowed(processor, ["acx"] * batch_size, library, prompts) == [
            {0}
        ] * batch_size
        ids = [
            prompt + [TOKENS[c] for c in "acx"] + [0] for prompt in prompts
        ]
        if library == "torch":
            logits = processor(torch.tensor(ids), torch.zeros(batch_size, VOCAB_SIZE))
            assert torch.isfinite(logits[:, 0]).all()
        else:
            logits = processor(
                np.array(ids), np.zeros((batch_size, VOCAB_SIZE), dtype=np.float32)
            )
            assert np.isfinite(logits[:, 0]).all()


@pytest.mark.parametrize(
    "seed,regex", [(1, "acx|bdy"), (2, "acx|bcy"), (8, "acx|bcy")]
)
@pytest.mark.parametrize("num_beams", [1, 2])
def test_transformers_beam_generation(seed, regex, num_beams):
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    import outlines

    tokenizers = pytest.importorskip("tokenizers")
    vocab = {
        s: i for i, s in enumerate(
            ["<eos>", "<pad>", "<unk>", "P", "a", "b", "c", "d", "x", "y"]
        )
    }
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.decoder = tokenizers.decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(vocab), n_positions=32, n_embd=16, n_layer=1,
            n_head=2, eos_token_id=0, pad_token_id=1, bos_token_id=3,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        )).cpu().eval()
    model = outlines.from_transformers(model, tokenizer)
    generator = outlines.Generator(
        model, outlines.types.Regex(regex), backend="outlines_core"
    )
    output = generator(
        "P", num_beams=num_beams, do_sample=False, max_new_tokens=4
    )
    assert re.fullmatch(regex, output)
