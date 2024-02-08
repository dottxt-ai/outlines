import pytest

from outlines.fsm.fsm import CFGFSM, RegexFSM, StopAtEosFSM


def test_stop_at_eos():
    class MockTokenizer:
        vocabulary = {"a": 1, "eos": 2}
        eos_token_id = 2

    fsm = StopAtEosFSM(MockTokenizer())

    assert fsm.allowed_token_ids(fsm.first_state) == [1, 2]
    assert fsm.allowed_token_ids(fsm.final_state) == [2]
    assert fsm.next_state(fsm.first_state, 2) == fsm.final_state
    assert fsm.next_state(fsm.first_state, 1) == fsm.first_state
    assert fsm.is_final_state(fsm.first_state) is False
    assert fsm.is_final_state(fsm.final_state) is True


def test_regex_vocabulary_error():
    class MockTokenizer:
        vocabulary = {"a": 1}
        special_tokens = {"eos"}
        eos_token_id = 3

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
    assert fsm.next_state(state=0, token_id=tokenizer.eos_token_id) == fsm.final_state

    assert fsm.is_final_state(0) is False
    assert fsm.is_final_state(fsm.final_state) is True


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

    assert set(fsm.allowed_token_ids(state=fsm.first_state)) == {1, 3, 5}
    state = fsm.next_state(state=fsm.first_state, token_id=1)
    assert fsm.generation == "{"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 2, 3}
    state = fsm.next_state(state=state, token_id=3)
    assert fsm.generation == "{["
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 3, 4}
    state = fsm.next_state(state=state, token_id=4)
    assert fsm.generation == "{[]"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {2}
    state = fsm.next_state(state=state, token_id=2)
    assert fsm.generation == "{[]}"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {5}
    state = fsm.next_state(state=state, token_id=5)
    assert fsm.generation == "{[]}"
    assert fsm.is_final_state(state)


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

    assert set(fsm.allowed_token_ids(state=fsm.first_state)) == {1}
    state = fsm.next_state(state=fsm.first_state, token_id=1)
    assert fsm.generation == "("
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 2}
    state = fsm.next_state(state=state, token_id=2)
    assert fsm.generation == "()"
    assert not fsm.is_final_state(state)

    # possible to continue or terminate
    assert set(fsm.allowed_token_ids(state=state)) == {1, 3}
    state = fsm.next_state(state=state, token_id=3)  # feed eos
    assert fsm.generation == "()"
    assert fsm.is_final_state(state)

    # once eos generated, can only terminate
    assert set(fsm.allowed_token_ids(state=state)) == {3}


def test_cfg_ignore_directive():
    class MockTokenizer:
        vocabulary = {"a": 1, " ": 2, "eos": 3}
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
        start: LETTER+
        LETTER: "a"
        WS: " "
        %ignore WS
    """
    tokenizer = MockTokenizer()
    fsm = CFGFSM(cfg_str, tokenizer)

    state = 0

    assert set(fsm.allowed_token_ids(state=0)) == {1, 2}
    state = fsm.next_state(state=0, token_id=2)
    assert fsm.generation == " "
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=0)) == {1, 2}
    state = fsm.next_state(state=0, token_id=1)
    assert fsm.generation == " a"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 2, 3}
    state = fsm.next_state(state=state, token_id=2)
    assert fsm.generation == " a "
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 2, 3}
    state = fsm.next_state(state=state, token_id=2)
    assert fsm.generation == " a  "
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 2, 3}
    state = fsm.next_state(state=state, token_id=1)
    assert fsm.generation == " a  a"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 2, 3}
    state = fsm.next_state(state=state, token_id=3)
    assert fsm.generation == " a  a"
    assert fsm.is_final_state(state)

    # once eos generated, can only terminate
    assert set(fsm.allowed_token_ids(state=state)) == {3}


def test_cfg_multitoken_terminal():
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

    assert set(fsm.allowed_token_ids(state=fsm.first_state)) == {1, 2}
    assert fsm.reset_state  # starting new regex
    state = fsm.next_state(state=fsm.first_state, token_id=1)
    assert fsm.generation == "a"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1}
    assert not fsm.reset_state  # continuing current regex
    state = fsm.next_state(state=state, token_id=1)
    assert fsm.generation == "aa"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {3}
    assert not fsm.reset_state  # completing current regex
    state = fsm.next_state(state=state, token_id=3)
    assert fsm.generation == "aa"
    assert fsm.is_final_state(state)


def test_cfg_allow_both_extend_and_shift_terminal():
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
    fsm = CFGFSM(cfg_str, tokenizer)

    assert set(fsm.allowed_token_ids(state=fsm.first_state)) == {1, 3}
    state = fsm.next_state(state=fsm.first_state, token_id=1)
    assert fsm.generation == "("
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {1, 3}
    state = fsm.next_state(state=state, token_id=3)
    assert fsm.generation == "(a"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {2, 3}
    state = fsm.next_state(state=state, token_id=3)
    assert fsm.generation == "(aa"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {2, 3}
    state = fsm.next_state(state=state, token_id=2)
    assert fsm.generation == "(aa)"
    assert not fsm.is_final_state(state)

    assert set(fsm.allowed_token_ids(state=state)) == {4}
    state = fsm.next_state(state=state, token_id=4)
    assert fsm.generation == "(aa)"
    assert fsm.is_final_state(state)
