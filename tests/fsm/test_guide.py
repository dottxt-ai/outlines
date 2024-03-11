import pytest

from outlines.fsm.guide import CFGGuide, Generate, RegexGuide, StopAtEOSGuide, Write


def test_stop_at_eos():
    class MockTokenizer:
        vocabulary = {"a": 1, "eos": 2}
        eos_token_id = 2

    fsm = StopAtEOSGuide(MockTokenizer())

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert instruction.tokens == [1, 2]

    instruction = fsm.get_next_instruction(fsm.final_state)
    assert isinstance(instruction, Write)
    assert instruction.tokens == [2]

    assert fsm.get_next_state(fsm.start_state, 2) == fsm.final_state
    assert fsm.get_next_state(fsm.start_state, 1) == fsm.start_state
    assert fsm.is_final_state(fsm.start_state) is False
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
        RegexGuide(regex_str, MockTokenizer())


def test_regex():
    class MockTokenizer:
        vocabulary = {"1": 1, "a": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"
    tokenizer = MockTokenizer()
    fsm = RegexGuide(regex_str, tokenizer)

    assert fsm.states_to_token_maps == {0: {1: 1}}

    instruction = fsm.get_next_instruction(0)
    assert isinstance(instruction, Generate)
    assert instruction.tokens == [1]

    assert fsm.get_next_state(state=0, token_id=1) == 1
    assert fsm.get_next_state(state=0, token_id=tokenizer.eos_token_id) == -1

    assert fsm.is_final_state(0) is False

    for state in fsm.final_states:
        assert fsm.is_final_state(state) is True


def test_regex_final_state():
    """Make sure that the FSM stays in the final state as we keep generating"""

    class MockTokenizer:
        vocabulary = {"`": 101, ".": 102, "\n": 103, "eos": 104}
        special_tokens = {"eos"}
        eos_token_id = 104

        def convert_token_to_string(self, token):
            return token

    regex_str = r"`\n(\.\n)?`\n"
    tokenizer = MockTokenizer()
    fsm = RegexGuide(regex_str, tokenizer)

    state = fsm.get_next_state(state=4, token_id=103)
    assert state == 5
    assert fsm.is_final_state(state)

    state = fsm.get_next_state(state=5, token_id=103)
    assert state == 5

    assert fsm.is_final_state(-1)


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
    fsm = CFGGuide(cfg_str, tokenizer)

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 3, 5}
    state = fsm.get_next_state(state=fsm.start_state, token_id=1)
    assert fsm.generation == "{"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2, 3}
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == "{["
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 3, 4}
    state = fsm.get_next_state(state=state, token_id=4)
    assert fsm.generation == "{[]"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {2}
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == "{[]}"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert set(instruction.tokens) == {5}
    state = fsm.get_next_state(state=state, token_id=5)
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
    fsm = CFGGuide(cfg_str, tokenizer)

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1}
    state = fsm.get_next_state(state=fsm.start_state, token_id=1)
    assert fsm.generation == "("
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2}
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == "()"
    assert not fsm.is_final_state(state)

    # possible to continue or terminate
    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 3}
    state = fsm.get_next_state(state=state, token_id=3)  # feed eos
    assert fsm.generation == "()"
    assert fsm.is_final_state(state)

    # once eos generated, can only terminate
    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert set(instruction.tokens) == {3}


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
    fsm = CFGGuide(cfg_str, tokenizer)

    state = 0

    instruction = fsm.get_next_instruction(0)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2}
    state = fsm.get_next_state(state=0, token_id=2)
    assert fsm.generation == " "
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(0)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2}
    state = fsm.get_next_state(state=0, token_id=1)
    assert fsm.generation == " a"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2, 3}
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == " a "
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2, 3}
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == " a  "
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2, 3}
    state = fsm.get_next_state(state=state, token_id=1)
    assert fsm.generation == " a  a"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2, 3}
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == " a  a"
    assert fsm.is_final_state(state)

    # once eos generated, can only terminate
    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert set(instruction.tokens) == {3}


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
    fsm = CFGGuide(cfg_str, tokenizer)

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 2}
    assert fsm.reset_state  # starting new regex
    state = fsm.get_next_state(state=fsm.start_state, token_id=1)
    assert fsm.generation == "a"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1}
    assert not fsm.reset_state  # continuing current regex
    state = fsm.get_next_state(state=state, token_id=1)
    assert fsm.generation == "aa"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert set(instruction.tokens) == {3}
    assert not fsm.reset_state  # completing current regex
    state = fsm.get_next_state(state=state, token_id=3)
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
    fsm = CFGGuide(cfg_str, tokenizer)

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 3}
    state = fsm.get_next_state(state=fsm.start_state, token_id=1)
    assert fsm.generation == "("
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {1, 3}
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == "(a"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {2, 3}
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == "(aa"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert set(instruction.tokens) == {2, 3}
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == "(aa)"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert set(instruction.tokens) == {4}
    state = fsm.get_next_state(state=state, token_id=4)
    assert fsm.generation == "(aa)"
    assert fsm.is_final_state(state)
