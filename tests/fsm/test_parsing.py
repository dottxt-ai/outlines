import importlib
from copy import copy

import lark.lark
import pytest
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken

from outlines.fsm.parsing import PartialLark, PartialPythonIndenter


@pytest.fixture
def cleanup_lark_import():
    yield
    # Clean up lark.lark.LarkOptions._defaults
    importlib.reload(lark.lark)


def test_partial_parsing(cleanup_lark_import):
    lp = PartialLark.open_from_package(
        "tests",
        "partial_python.lark",
        ["fsm"],
        parser="lalr",
        postlex=PartialPythonIndenter(),
        start="file_input",
        deterministic=True,
    )

    # End with a potentially unfinished NAME
    parser_state = lp.parse("x")
    assert parser_state.state_stack == [0]
    last_token = parser_state.lexer.state.last_token
    assert last_token.type == "partial"
    assert last_token.value.fsm_state_seq == (0, 15)
    assert last_token.value.is_not_finished is True
    assert not parser_state.value_stack

    # End with an ignored token
    parser_state = lp.parse("x ")
    assert parser_state.state_stack == [0, 692]
    last_token = parser_state.lexer.state.last_token
    assert last_token.type == "partial"
    assert last_token.value.fsm_state_seq == (0, 1)
    assert last_token.value.is_not_finished is True
    assert not parser_state.value_stack

    # Could be a complete `=` or the start of a `==`
    parser_state = lp.parse("x =")
    assert parser_state.state_stack == [0, 692]
    last_token = parser_state.lexer.state.last_token
    assert last_token.type == "partial"
    assert any(
        term_info.terminal_name == "EQUAL"
        for term_info in last_token.value.terminals_and_info
    )
    assert not parser_state.value_stack

    parser_state = lp.parse("x = '")
    assert parser_state.state_stack == [0, 58, 59]
    last_token = parser_state.lexer.state.last_token
    assert last_token.type == "partial"
    assert last_token.value.fsm_state_seq == (0, 6)
    assert last_token.value.is_not_finished is True
    assert not parser_state.value_stack

    parser_state = lp.parse("x = 'hi")
    assert parser_state.state_stack == [0, 58, 59]
    last_token = parser_state.lexer.state.last_token
    assert last_token.type == "partial"
    assert last_token.value.fsm_state_seq == (0, 6, 6, 6)
    assert last_token.value.is_not_finished is True
    assert not parser_state.value_stack

    parser_state = lp.parse("x = ('hi")
    assert parser_state.state_stack == [0, 58, 59, 254]
    last_token = parser_state.lexer.state.last_token
    assert last_token.type == "partial"
    assert last_token.value.fsm_state_seq == (0, 6, 6, 6)
    assert last_token.value.is_not_finished is True
    assert not parser_state.value_stack

    parser_state = lp.parse("def")
    assert parser_state.state_stack == [0]
    last_token = parser_state.lexer.state.last_token
    assert last_token.type == "partial"
    assert last_token.value.fsm_state_seq == (0, 26, 99, 100)
    assert last_token.value.is_not_finished is True
    assert not parser_state.value_stack

    # Now, try something incremental
    last_lexer_state = parser_state.lexer.state
    last_lexer_state.text += " blah()"
    lp.parse_from_state(parser_state, is_end=False)
    last_token = parser_state.lexer.state.last_token
    assert not parser_state.value_stack

    last_lexer_state = parser_state.lexer.state
    last_valid_token = last_lexer_state.last_token
    assert last_valid_token.type == "RPAR"
    assert not parser_state.value_stack

    # Something incremental and a little more complicated
    parser_state = lp.parse("x = 1\ndef foo(x):\n  ")
    assert parser_state.state_stack == [0, 94, 600, 601, 602, 607, 608, 269]
    last_lexer_state = parser_state.lexer.state
    last_lexer_state.text += "  return x"

    lp.parse_from_state(parser_state, is_end=False)
    assert parser_state.state_stack == [
        0,
        94,
        600,
        601,
        602,
        607,
        608,
        269,
        764,
        95,
        305,
    ]
    last_token = parser_state.lexer.state.last_token
    assert last_token.type == "partial"

    with pytest.raises(UnexpectedToken):
        lp.parse("def \n")

    with pytest.raises(UnexpectedToken):
        lp.parse("def hot no")

    lp = PartialLark.open_from_package(
        "tests",
        "partial_python.lark",
        ["fsm"],
        parser="lalr",
        postlex=PartialPythonIndenter(),
        start="file_input",
        use_value_stack=True,
    )
    parser_state = lp.parse("x = ('hi")
    lp.parse_from_state(parser_state, is_end=False)
    assert len(parser_state.state_stack) == 4
    assert parser_state.value_stack[-1].type == "LPAR"


def test_sequential_parse_example(cleanup_lark_import):
    input_tokens = [
        "x ",
        "= ",
        "1",
        "\nde",
        "f ",
        "foo(",
        "x)",
        ":\n",
        "  ",
        "  return x",
        " + 1",
        "\n",
        "z ",
        "= ",
        "foo(",
        '"hi',
        '")\n',
    ]
    vocab = sorted(set(input_tokens))

    lp = PartialLark.open_from_package(
        "tests",
        "partial_python.lark",
        ["fsm"],
        parser="lalr",
        postlex=PartialPythonIndenter(),
        start="file_input",
        deterministic=True,
    )
    parser_state = lp.parse("")

    token_seq = ""
    for i, token in enumerate(input_tokens):
        token_seq += token

        lex_state = parser_state.lexer.state
        lex_state.text = token_seq

        lp.parse_from_state(parser_state, is_end=False)

        next_vocab = set()
        for test_token in vocab:
            ps = copy(parser_state)
            ls = ps.lexer.state
            ls.text = token_seq + test_token

            if i + 1 < len(input_tokens) and test_token == input_tokens[i + 1]:
                lp.parse_from_state(ps, is_end=False)
                next_vocab.add(test_token)
            else:
                try:
                    lp.parse_from_state(ps, is_end=False)
                    next_vocab.add(test_token)
                except (EOFError, UnexpectedToken, UnexpectedCharacters, DedentError):
                    pass

        if i + 1 == len(input_tokens):
            assert all(tk in next_vocab for tk in ["\n", "\nde", "  ", " + 1"])


# TODO: Remove once fsm_union and walk_fsm are implemented in Outlines-Core
import interegular  # noqa

from outlines.fsm.parsing import fsm_union, walk_fsm  # noqa


def test_outlines_interegular_union_consistency():
    fsm0 = interegular.parse_pattern(r"abc").to_fsm()
    fsm1 = interegular.parse_pattern(r"WXYZ").to_fsm()
    fsm2 = interegular.parse_pattern(r"12345").to_fsm()

    interegular_unioned_fsm = fsm0 | fsm1 | fsm2
    outlines_unioned_fsm, _ = fsm_union([fsm0, fsm1, fsm2])

    assert list(outlines_unioned_fsm.strings()) == list(
        interegular_unioned_fsm.strings()
    )


def _reconstruct_fsms(fsm, fsms_to_trans_finals):
    """Reconstruct the original fsms for testing purposes"""
    reconstructed_fsms = []
    for transitions, finals, state_map in fsms_to_trans_finals.values():
        inv_state_map = {new: orig for orig, news in state_map.items() for new in news}
        states = set(inv_state_map.values())
        initial = inv_state_map.get(fsm.initial) or next(
            (orig for orig, news in state_map.items() if fsm.initial in news), None
        )
        finals = {inv_state_map[s] for s in finals}

        transition_map = {}
        alphabet = {}
        for trans_id, (from_state, to_state) in enumerate(transitions):
            orig_from, orig_to = inv_state_map[from_state], inv_state_map[to_state]
            # Collect symbols associated with the transition
            symbols = {
                symbol
                for trans, dest in fsm.map.get(from_state, {}).items()
                if dest == to_state
                for symbol in fsm.alphabet.by_transition.get(trans, [])
            }
            if symbols:
                # NOTE: THIS RECONSTRUCTOR DOESN'T WORK FOR MORE THAN ONE TRANSITION PER SYMBOL
                assert len(symbols) == 1
                symbol = list(symbols)[0]
                alphabet[symbol] = trans_id
                transition_map.setdefault(orig_from, {})[trans_id] = orig_to

        reconstructed_fsms.append(
            interegular.fsm.FSM(
                alphabet=interegular.fsm.Alphabet(alphabet),
                states=frozenset(states),
                initial=initial,
                finals=frozenset(finals),
                map=transition_map,
                __no_validation__=True,
            )
        )
    return reconstructed_fsms


def test_fsm_to_trans_finals_reconstruction():
    """Assert that _fsms_to_trans_finals is correct by reconstructing original fsms"""
    fsm0 = interegular.parse_pattern(r"abc").to_fsm()
    fsm1 = interegular.parse_pattern(r"XYZ").to_fsm()
    fsm2 = interegular.parse_pattern(r"12345").to_fsm()

    fsm, _fsms_to_trans_finals = fsm_union([fsm0, fsm1, fsm2])

    reconstructed = _reconstruct_fsms(fsm, _fsms_to_trans_finals)

    # assert reconstruction equivalent
    assert list(fsm0.strings()) == list(reconstructed[0].strings())
    assert list(fsm1.strings()) == list(reconstructed[1].strings())
    assert list(fsm2.strings()) == list(reconstructed[2].strings())


def test_walk_fsm():
    fsm = interegular.parse_pattern(r"abc*d").to_fsm()
    # convert to BetterFSM
    fsm = fsm_union([fsm])[0]

    # if match, produce equivalent number of states, assert state can terminate
    transitions = [fsm.alphabet[letter] for letter in "abcccd"]
    accepted_states = walk_fsm(fsm, transitions, fsm.initial, full_match=True)
    assert len(accepted_states) == len(transitions)
    assert accepted_states[-1] in fsm.finals

    # if no match, assert empty
    accepted_states = walk_fsm(
        fsm, [fsm.alphabet[letter] for letter in "b"], fsm.initial, full_match=True
    )
    assert accepted_states == []

    # if full_match, but last state not present, assert empty
    accepted_states = walk_fsm(
        fsm, [fsm.alphabet[letter] for letter in "abc"], fsm.initial, full_match=True
    )
    assert accepted_states == []
