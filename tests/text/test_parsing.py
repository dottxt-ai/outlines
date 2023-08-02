import random
import re
from copy import copy

import interegular
import pytest
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken

from outlines.text.parsing import (
    PartialLark,
    PartialPythonIndenter,
    find_partial_matches,
    fsm_union,
    get_sub_fsms_from_seq,
    make_deterministic_fsm,
    map_partial_states_to_vocab,
    terminals_to_fsms,
)


def test_partial_parsing():
    lp = PartialLark.open_from_package(
        "tests",
        "partial_python.lark",
        ["text"],
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

    lp = PartialLark.open_from_package(
        "tests",
        "partial_python.lark",
        ["text"],
        parser="lalr",
        postlex=PartialPythonIndenter(),
        start="file_input",
        use_value_stack=True,
    )
    parser_state = lp.parse("x = ('hi")
    lp.parse_from_state(parser_state, is_end=False)
    assert len(parser_state.state_stack) == 4
    assert parser_state.value_stack[-1].type == "LPAR"


def test_sequential_parse_example():
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
        ["text"],
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


def test_find_partial_matches():
    name_pattern = interegular.parse_pattern(r"[^\W\d]\w*")
    name_fsm, _ = make_deterministic_fsm(name_pattern.to_fsm().reduce())
    assert name_fsm.initial == 0

    def_pattern = interegular.parse_pattern("def")
    def_fsm, _ = make_deterministic_fsm(def_pattern.to_fsm().reduce())
    assert def_fsm.initial == 0

    assert find_partial_matches(def_fsm, "def") == {(2, (0, 1, 2, 3))}
    assert find_partial_matches(def_fsm, "de") == {(1, (0, 1, 2))}
    assert find_partial_matches(def_fsm, "d") == {(0, (0, 1))}
    assert find_partial_matches(def_fsm, "") == set()
    assert find_partial_matches(def_fsm, "df") == set()
    assert find_partial_matches(def_fsm, "ef") == {(1, (1, 2, 3))}
    assert find_partial_matches(def_fsm, "e") == {(0, (1, 2))}
    assert find_partial_matches(def_fsm, "f") == {(0, (2, 3))}
    assert find_partial_matches(def_fsm, "ef foo") == {(1, (1, 2, 3))}

    # This string has a `DEF` token in it, but should ultimately not lex one
    assert find_partial_matches(def_fsm, "defb") == {(2, (0, 1, 2, 3))}

    # `NAME` can have multiple start states for this input
    assert find_partial_matches(name_fsm, "d") == {
        (0, (0, 1)),
        (0, (1, 1)),
    }
    # Not this case
    assert find_partial_matches(name_fsm, "1d") == {(1, (1, 1, 1))}

    assert find_partial_matches(name_fsm, "blah") == {
        (3, (0, 1, 1, 1, 1)),
        (3, (1, 1, 1, 1, 1)),
    }

    float_pattern = interegular.parse_pattern(
        r"([+-]?((0|[1-9]+)([.][0-9]*)?)|([.][0-9]+))"
    )
    float_fsm, _ = make_deterministic_fsm(float_pattern.to_fsm().reduce())
    assert 5 in float_fsm.finals
    assert 2 not in float_fsm.finals

    res = find_partial_matches(float_fsm, ".")
    assert res == {(0, (3, 5)), (0, (4, 5)), (0, (0, 2))}

    joins_fsm, _ = make_deterministic_fsm(
        interegular.parse_pattern(r"(JOIN LEFT|JOIN)").to_fsm().reduce()
    )
    res = find_partial_matches(
        joins_fsm, "JOIN BLAH", joins_fsm.initial, full_match=False
    )
    assert res == {(3, (0, 1, 2, 3, 4))}

    res = find_partial_matches(joins_fsm, "JOIN L", joins_fsm.initial, full_match=False)
    assert res == {(5, (0, 1, 2, 3, 4, 5, 6))}

    res = find_partial_matches(joins_fsm, "JOI", joins_fsm.initial, full_match=False)
    assert res == {(2, (0, 1, 2, 3))}


def test_map_partial_states_to_vocab_python():
    pyparser = PartialLark.open_from_package(
        "tests",
        "partial_python.lark",
        ["text"],
        parser="lalr",
        postlex=PartialPythonIndenter(),
        start="file_input",
    )

    symbol_names_and_fsms = terminals_to_fsms(pyparser)
    test_symbols = {"DEF", "NAME", "__IGNORE_0"}
    symbol_names_and_fsms = {
        k: v for k, v in symbol_names_and_fsms.items() if k in test_symbols
    }

    assert len(symbol_names_and_fsms["DEF"].states) == 4
    assert len(symbol_names_and_fsms["NAME"].states) == 2
    assert len(symbol_names_and_fsms["__IGNORE_0"].states) == 2

    vocabulary = ["d", "e", "ef foo", "f ", " ", "1d", "<EOS>"]

    pstate_to_vocab, possible_paths = map_partial_states_to_vocab(
        vocabulary, symbol_names_and_fsms
    )

    assert dict(pstate_to_vocab) == {
        ("__IGNORE_0", 0): {4},
        ("__IGNORE_0", 1): {4},
        ("NAME", 0): {0, 1, 2, 3},
        ("NAME", 1): {0, 1, 2, 3, 5},
        ("DEF", 0): {0},
        ("DEF", 1): {1, 2},
        ("DEF", 2): {3},
    }
    assert possible_paths["__IGNORE_0"] == {0: {1}, 1: {1}}
    assert possible_paths["NAME"] == {0: {1}, 1: {1}}
    assert possible_paths["DEF"] == {0: {1}, 1: {2, 3}, 2: {3}}

    pstate_to_vocab, possible_paths = map_partial_states_to_vocab(
        vocabulary, symbol_names_and_fsms, final_state_string="<EOS>"
    )

    assert dict(pstate_to_vocab) == {
        ("__IGNORE_0", 0): {
            4,
        },
        ("__IGNORE_0", 1): {4, 6},
        ("NAME", 0): {0, 1, 2, 3},
        ("NAME", 1): {0, 1, 2, 3, 5, 6},
        ("DEF", 0): {
            0,
        },
        ("DEF", 1): {1, 2},
        ("DEF", 2): {
            3,
        },
        ("DEF", 3): {
            6,
        },
    }
    assert possible_paths["__IGNORE_0"] == {0: {1}, 1: {1}}
    assert possible_paths["NAME"] == {0: {1}, 1: {1}}
    assert possible_paths["DEF"] == {0: {1}, 1: {2, 3}, 2: {3}}


def test_map_partial_states_to_vocab_regex():
    regex_string = r"([0-9]+([.][0-9]*)?|[.][0-9]+)"
    regex_pattern = interegular.parse_pattern(regex_string)
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    vocabulary = [
        "1.",
        "2",
        "3.",
        ".",
        ".80",
        "42",
        "1a",
        " ",
        "0",
        "a",
        "b",
        "$",
        "<EOS>",
    ]

    # We want the vocabulary strings to entirely match the regex--not just the
    # prefixes of the vocabulary strings
    def partial_match_filter(string, end_idx, state_seq):
        if end_idx is not None and end_idx < len(string) - 1:
            return False
        return True

    pstate_to_vocab, possible_paths = map_partial_states_to_vocab(
        vocabulary, {"FLOAT": regex_fsm}, partial_match_filter, "<EOS>"
    )

    assert sorted(pstate_to_vocab.values(), key=lambda x: -len(x)) == [
        {0, 1, 2, 3, 4, 5, 8, 12},
        {0, 1, 2, 3, 4, 5, 8},
        {1, 5, 8, 12},
        {1, 5, 8},
    ]
    assert possible_paths["FLOAT"] == {2: {2, 3}, 0: {1, 2, 3}, 3: {3}, 1: {3}}

    pstate_to_vocab = {k: tuple(v) for k, v in pstate_to_vocab.items()}

    random.seed(24080)

    for n in range(50):
        # Start at the initial state
        pstate = ("FLOAT", regex_fsm.initial)

        sample_seq = ""

        for i in range(5):
            next_support = pstate_to_vocab[pstate]

            (next_sample_idx,) = random.sample(next_support, 1)

            next_sample = vocabulary[next_sample_idx]

            if next_sample == "<EOS>":
                break

            sample_seq += next_sample

            # Continue matching from where we left off
            (pmatch,) = find_partial_matches(
                regex_fsm, next_sample, start_state=pstate[-1]
            )

            # Create the next state
            pstate = (pstate[0], pmatch[1][-1])

            # TODO: We could check if the FSM is done (i.e. in an final/accept
            # state) and end the sampling loop

        # Make sure the whole thing matches the regex
        assert re.fullmatch(regex_string, sample_seq) is not None


def test_get_sub_fsms_from_seq():
    name_pattern = interegular.parse_pattern(r"[^\W\d]\w*")
    name_fsm, _ = make_deterministic_fsm(name_pattern.to_fsm().reduce())

    def_pattern = interegular.parse_pattern("def")
    def_fsm, _ = make_deterministic_fsm(def_pattern.to_fsm().reduce())

    match_pattern = interegular.parse_pattern("match")
    match_fsm, _ = make_deterministic_fsm(match_pattern.to_fsm().reduce())

    peq_pattern = interegular.parse_pattern(r"\+=")
    peq_fsm, _ = make_deterministic_fsm(peq_pattern.to_fsm().reduce())

    plus_pattern = interegular.parse_pattern(r"\+")
    plus_fsm, _ = make_deterministic_fsm(plus_pattern.to_fsm().reduce())

    fsms = [def_fsm, match_fsm, name_fsm, peq_fsm, plus_fsm]

    fsm, fsms_to_trans_finals = fsm_union(fsms)

    assert fsms_to_trans_finals == {
        0: ({(0, 3), (3, 9), (9, 10)}, {10}, {0: {0}, 1: {3}, 2: {9}, 3: {10}}),
        1: (
            {(0, 4), (4, 5), (5, 6), (6, 7), (7, 8)},
            {8},
            {0: {0}, 1: {4}, 2: {5}, 3: {6}, 4: {7}, 5: {8}},
        ),
        2: (
            {
                (0, 2),
                (0, 3),
                (0, 4),
                (2, 2),
                (3, 2),
                (3, 9),
                (4, 2),
                (4, 5),
                (5, 2),
                (5, 6),
                (6, 2),
                (6, 7),
                (7, 2),
                (7, 8),
                (8, 2),
                (9, 2),
                (9, 10),
                (10, 2),
            },
            {2, 3, 4, 5, 6, 7, 8, 9, 10},
            {0: {0}, 1: {2, 3, 4, 5, 6, 7, 8, 9, 10}},
        ),
        3: ({(0, 1), (1, 11)}, {11}, {0: {0}, 1: {1}, 2: {11}}),
        4: ({(0, 1)}, {1}, {0: {0}, 1: {1}}),
    }

    assert not fsm.accepts("1a")
    assert fsm.accepts("a1")
    assert fsm.accepts("def")
    assert fsm.accepts("match")
    assert fsm.accepts("+=")
    assert fsm.accepts("+")

    ((_, state_seq),) = find_partial_matches(fsm, "def", start_state=fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, False, True), (2, True, True)]

    # Make sure the old-to-new state map is correct
    ((_, def_state_seq),) = find_partial_matches(
        def_fsm, "def", start_state=fsm.initial
    )
    def_old_to_new_states = fsms_to_trans_finals[0][2]
    assert all(
        new_state in def_old_to_new_states[old_state]
        for old_state, new_state in zip(def_state_seq, state_seq)
    )

    ((_, state_seq),) = find_partial_matches(fsm, "ef", start_state=fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(2, True, True)]

    ((_, name_state_seq),) = find_partial_matches(
        name_fsm, "ef", start_state=fsm.initial
    )
    name_old_to_new_states = fsms_to_trans_finals[2][2]
    assert all(
        new_state in name_old_to_new_states[old_state]
        for old_state, new_state in zip(name_state_seq, state_seq)
    )

    ((_, state_seq),) = find_partial_matches(fsm, "match", start_state=fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(1, False, True), (2, True, True)]

    ((_, match_state_seq),) = find_partial_matches(
        match_fsm, "match", start_state=fsm.initial
    )
    match_old_to_new_states = fsms_to_trans_finals[1][2]
    assert all(
        new_state in match_old_to_new_states[old_state]
        for old_state, new_state in zip(match_state_seq, state_seq)
    )

    ((_, state_seq),) = find_partial_matches(fsm, "defa", start_state=fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(2, True, True)]

    ((_, state_seq),) = find_partial_matches(fsm, "de", start_state=fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, True, False), (2, True, True)]

    ((_, state_seq),) = find_partial_matches(fsm, "+", start_state=fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(3, True, False), (4, False, True)]

    ((_, state_seq),) = find_partial_matches(fsm, "+=", start_state=fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(3, False, True)]

    # Test some overlapping patterns
    join_fsms = [
        interegular.parse_pattern(r"JOIN").to_fsm().reduce(),
        interegular.parse_pattern(r"JOIN LEFT").to_fsm().reduce(),
    ]
    fsm, fsms_to_trans_finals = fsm_union(join_fsms)
    ((_, state_seq),) = find_partial_matches(
        fsm, "OI", start_state=None, full_match=False
    )
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, True, False), (1, True, False)]

    ((_, state_seq),) = find_partial_matches(
        fsm, "N", start_state=None, full_match=False
    )
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, False, True), (1, True, False)]

    ((_, state_seq),) = find_partial_matches(
        fsm, " ", start_state=None, full_match=False
    )
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(1, True, False)]
