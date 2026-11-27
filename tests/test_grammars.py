import json

import llguidance
import pytest

from outlines import grammars


class ByteTokenizer:
    """One token per byte, so a string is accepted iff the grammar accepts it."""

    eos_token_id = 256
    bos_token_id = None
    tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]
    special_token_ids = [256]

    def __call__(self, s):
        return list(s)


def grammar_accepts(grammar: str, text: str) -> bool:
    tokenizer = llguidance.LLTokenizer(llguidance.TokenizerWrapper(ByteTokenizer()))
    matcher = llguidance.LLMatcher(tokenizer, llguidance.grammar_from("lark", grammar))
    tokens = list(text.encode())
    return matcher.try_consume_tokens(tokens) == len(tokens) and matcher.is_accepting()


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '[1, -2.5, 3e10, 0.5E-3]',
        '{"a": [true, false, null], "b": {}}',
        r'"a\"b\\c\/d\né"',
        '0',
    ],
)
def test_json_grammar_accepts_valid_json(text):
    json.loads(text)
    assert grammar_accepts(grammars.json, text)


@pytest.mark.parametrize(
    "text",
    [".5", "1.", "+1", "01", "-", "1e", r'"\q"', r'"\u00g9"', '"\x01"'],
)
def test_json_grammar_rejects_invalid_json(text):
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    assert not grammar_accepts(grammars.json, text)
