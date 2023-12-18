import pytest

from outlines.fsm.fsm import CFGFSM, RegexFSM, StopAtTokenFSM


def test_stop_at_token():
    class MockTokenizer:
        vocabulary = {"a": 1, "eos": 2}
        special_tokens = {"eos"}

    fsm = StopAtTokenFSM(MockTokenizer(), 2)

    assert fsm.allowed_token_ids(0) == [1, 2]
    assert fsm.allowed_token_ids(1) == [2]
    assert fsm.next_state(0, 2) == 1
    assert fsm.next_state(0, 1) == 0
    assert fsm.is_final_state(0) is False
    assert fsm.is_final_state(1) is True


def test_regex_vocabulary_error():
    class MockTokenizer:
        vocabulary = {"a": 1}
        special_tokens = {"eos"}

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"

    with pytest.raises(ValueError, match="The vocabulary"):
        RegexFSM(regex_str, MockTokenizer())


def test_regex():
    class MockTokenizer:
        vocabulary = {"1": 1, "a": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"
    tokenizer = MockTokenizer()
    fsm = RegexFSM(regex_str, tokenizer)

    assert fsm.states_to_token_maps == {0: {1: 1}}
    assert fsm.allowed_token_ids(state=0) == [1]
    assert fsm.next_state(state=0, token_id=1) == 1
    assert fsm.next_state(state=0, token_id=tokenizer.eos_token_id) == -1

    assert fsm.is_final_state(1) is True
    assert fsm.is_final_state(0) is False
    assert fsm.is_final_state(-1) is True


def test_cfg():
    class MockTokenizer:
        vocabulary = {"{": 1, "}": 2, "[": 3, "]": 4, "eos": 5}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 5

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: expr
        expr: "{" expr "}" | "[" expr "]" |
    """
    tokenizer = MockTokenizer()
    fsm = CFGFSM(cfg_str, tokenizer)

    assert set(fsm.allowed_token_ids(state=0)) == {1, 3, 5}
    state = fsm.next_state(state=0, token_id=1)
    assert fsm.generations[0] == "{"
    assert not fsm.is_final_state(0)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 2, 3}
    state = fsm.next_state(state=state, token_id=3)
    assert fsm.generations[0] == "{["
    assert not fsm.is_final_state(0)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 3, 4}
    state = fsm.next_state(state=state, token_id=4)
    assert fsm.generations[0] == "{[]"
    assert not fsm.is_final_state(0)

    assert set(fsm.allowed_token_ids(state=state)) == {2}
    state = fsm.next_state(state=state, token_id=2)
    assert fsm.generations[0] == "{[]}"
    assert not fsm.is_final_state(0)

    assert set(fsm.allowed_token_ids(state=state)) == {5}
    state = fsm.next_state(state=state, token_id=5)
    assert fsm.generations[0] == "{[]}"
    assert fsm.is_final_state(0)


def test_cfg_early_termination():
    class MockTokenizer:
        vocabulary = {"(": 1, ")": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: expr+
        expr: "(" subexpr ")"
        subexpr: expr |
    """
    tokenizer = MockTokenizer()
    fsm = CFGFSM(cfg_str, tokenizer)

    assert set(fsm.allowed_token_ids(state=0)) == {1}
    state = fsm.next_state(state=0, token_id=1)
    assert fsm.generations[0] == "("
    assert not fsm.is_final_state(0)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 2}
    state = fsm.next_state(state=state, token_id=2)
    assert fsm.generations[0] == "()"
    assert not fsm.is_final_state(0)

    # possible to continue or terminate
    assert set(fsm.allowed_token_ids(state=state)) == {1, 3}
    state = fsm.next_state(state=state, token_id=3)  # feed eos
    assert fsm.generations[0] == "()"
    assert fsm.is_final_state(0)

    # once eos generated, can only terminate
    assert set(fsm.allowed_token_ids(state=state)) == {3}


def test_cfg_multitoken_subexpr():
    class MockTokenizer:
        vocabulary = {"a": 1, "b": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: S
        S: "aa" | "bb"
    """
    tokenizer = MockTokenizer()
    fsm = CFGFSM(cfg_str, tokenizer)

    assert set(fsm.allowed_token_ids(state=0)) == {1, 2}
    assert fsm.reset_state[0]  # starting new regex
    state = fsm.next_state(state=0, token_id=1)
    assert fsm.generations[0] == "a"
    assert not fsm.is_final_state(0)

    assert set(fsm.allowed_token_ids(state=state)) == {1}
    assert not fsm.reset_state[0]  # continuing current regex
    state = fsm.next_state(state=state, token_id=1)
    assert fsm.generations[0] == "aa"
    assert not fsm.is_final_state(0)

    assert set(fsm.allowed_token_ids(state=state)) == {3}
    assert not fsm.reset_state[0]  # completing current regex
    state = fsm.next_state(state=state, token_id=3)
    assert fsm.generations[0] == "aa"
    assert fsm.is_final_state(0)


@pytest.mark.xfail(
    strict=True,
    reason="Current regex implementation is not complete",
    raises=NotImplementedError,
)
def test_cfg_overlapping_subexpr():
    class MockTokenizer:
        vocabulary = {"a": 1, "b": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: S
        S: "a" | "b" | "aa" | "bb"
    """
    tokenizer = MockTokenizer()
    fsm = CFGFSM(cfg_str, tokenizer)

    assert set(fsm.allowed_token_ids(state=0)) == {1, 2}
    state = fsm.next_state(state=0, token_id=1)
    assert fsm.generations[0] == "a"
    assert not fsm.is_final_state(0)

    # INTENDED LOGIC
    # This will fail until we fix TODO raised in https://github.com/outlines-dev/outlines/pull/391
    try:
        assert set(fsm.allowed_token_ids(state=state)) == {1, 3}
    except AssertionError:
        raise NotImplementedError("TODO: fix this")

    # CURRENT LOGIC
    # For now, the FSM can only generate the greedy completion, ending at "a", never "aa"
    # This implementation is sound, and always terminates, but is not complete
    assert set(fsm.allowed_token_ids(state=state)) == {3}
    state = fsm.next_state(state=state, token_id=3)
    assert fsm.generations[0] == "a"
    assert fsm.is_final_state(0)
