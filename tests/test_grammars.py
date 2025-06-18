import pytest

import outlines.grammars as grammars
from outlines.fsm.guide import CFGGuide


@pytest.mark.parametrize("grammar", [grammars.json, grammars.arithmetic])
def test_grammar_module(grammar):
    class MockTokenizer:
        vocabulary = {"(": 1, ")": 2, "a": 3, "eos": 4}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 4

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: s
        s: "(" s ")" | /a+/
    """
    tokenizer = MockTokenizer()
    fsm = CFGGuide(cfg_str, tokenizer)
    assert isinstance(fsm, CFGGuide)
