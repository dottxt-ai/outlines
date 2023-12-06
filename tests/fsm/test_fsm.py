import pytest

from outlines.fsm.fsm import RegexFSM, StopAtTokenFSM


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
