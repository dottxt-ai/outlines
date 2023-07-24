import random
import re

import interegular
import pytest
from lark import Lark
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken

from outlines.text.parsing import (
    PartialPythonIndenter,
    copy_parser_state,
    create_pmatch_parser_states,
    find_partial_matches,
    map_partial_states_to_vocab,
    parse_to_end,
    terminals_to_fsms,
    terminals_to_lalr_states,
)


def test_parse_to_end():
    pyparser = Lark.open_from_package(
        "lark",
        "python.lark",
        ["grammars"],
        parser="lalr",
        postlex=PartialPythonIndenter(),
        start="file_input",
    )

    ip = pyparser.parse_interactive("x")
    parser_state = copy_parser_state(ip.parser_state)
    parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert not parser_state.value_stack
    assert expected_next_tokens == {"NAME"}

    ip = pyparser.parse_interactive("x = '")
    parser_state = copy_parser_state(ip.parser_state)
    parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert parser_state.value_stack[-1].type == "EQUAL"
    assert expected_next_tokens == {"LONG_STRING", "STRING"}

    ip = pyparser.parse_interactive("x = 'hi")
    parser_state = copy_parser_state(ip.parser_state)
    parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert parser_state.value_stack[-1].type == "EQUAL"
    assert expected_next_tokens == {"STRING"}

    ip = pyparser.parse_interactive("x = ('hi")
    parser_state = copy_parser_state(ip.parser_state)
    parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert parser_state.value_stack[-1].type == "LPAR"
    assert expected_next_tokens == {"STRING"}

    ip = pyparser.parse_interactive("def")
    parser_state = copy_parser_state(ip.parser_state)
    parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert not parser_state.value_stack
    assert expected_next_tokens == {"NAME", "DEF"}

    # Now, try something incremental
    parser_state = copy_parser_state(parser_state)
    last_lexer_state = parser_state.lexer.state
    last_lexer_state.text = "def blah()"

    (parser_state, expected_next_tokens) = parse_to_end(parser_state)

    last_lexer_state = parser_state.lexer.state
    last_valid_token = last_lexer_state.last_token
    assert last_valid_token.type == "RPAR"
    assert not expected_next_tokens


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
        '"hi' '")',
    ]
    vocab = set(input_tokens)

    pyparser = Lark.open_from_package(
        "lark",
        "python.lark",
        ["grammars"],
        parser="lalr",
        postlex=PartialPythonIndenter(),
        start="file_input",
    )
    ip = pyparser.parse_interactive("")
    parser_state = copy_parser_state(ip.parser_state)

    token_seq = ""
    for i, token in enumerate(input_tokens):
        token_seq += token

        lex_state = parser_state.lexer.state
        lex_state.text = token_seq

        parser_state, partial_tokens = parse_to_end(parser_state)

        next_vocab = set()
        for test_token in vocab:
            ps = copy_parser_state(parser_state)
            ls = ps.lexer.state
            ls.text = token_seq + test_token

            try:
                # TODO: The resulting states could possibly be reused?
                parse_to_end(ps)
                next_vocab.add(test_token)
            except (UnexpectedToken, UnexpectedCharacters, DedentError):
                pass

        if i + 1 < len(input_tokens):
            assert input_tokens[i + 1] in next_vocab
        else:
            assert all(tk in next_vocab for tk in ["\n", "\nde", "  ", " + 1"])


def test_partial_match():
    name_pattern = interegular.parse_pattern(r"[^\W\d]\w*")
    name_fsm = name_pattern.to_fsm().reduce()
    assert name_fsm.initial == 0

    def_pattern = interegular.parse_pattern("def")
    def_fsm = def_pattern.to_fsm().reduce()
    assert def_fsm.initial == 0

    assert find_partial_matches(def_fsm, "def") == {(2, (0, 1, 2, 3))}
    assert find_partial_matches(def_fsm, "de") == {(None, (0, 1, 2))}
    assert find_partial_matches(def_fsm, "d") == {(None, (0, 1))}
    assert find_partial_matches(def_fsm, "") == set()
    assert find_partial_matches(def_fsm, "df") == set()
    assert find_partial_matches(def_fsm, "ef") == {(1, (1, 2, 3))}
    assert find_partial_matches(def_fsm, "e") == {(None, (1, 2))}
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
    float_fsm = float_pattern.to_fsm().reduce()

    # XXX: It look like there's a lot of set/frozenset usage that prevents us
    # from adequately reproducing the exact state sequences in this case.
    # It seems to stem from `_CharGroup`s and the FSM map construction process.
    res = find_partial_matches(float_fsm, ".")
    assert {v[0] for v in res} == {0, 0, None}
    # Make sure that the terminated sequences actually end in final states
    assert all(v[1][-1] in float_fsm.finals for v in res if v[0] == 0)
    # Make sure that the non-terminated sequences don't end in final states
    assert all(v[1][-1] not in float_fsm.finals for v in res if v[0] != 0)


def test_map_partial_states_to_vocab_python():
    pyparser = Lark.open_from_package(
        "lark",
        "python.lark",
        ["grammars"],
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


def test_parse_from_partial_match():
    """Make sure we can continue parsing from an FSM-based partial match."""
    lp = Lark(
        r"""
start: funcdef

funcdef: "def" name "(" ")" ":" attr_pattern

attr_pattern: NAME ("." NAME)+ -> value

%ignore /[\t \f]+/  // WS

!name: NAME | "match" | "case"
NAME: /[^\W\d]\w*/


    """,
        parser="lalr",
        postlex=PartialPythonIndenter(),
    )

    terminals_to_states = terminals_to_lalr_states(lp)
    symbol_names_and_fsms = terminals_to_fsms(lp)

    term_type = "DEF"
    term_fsm = symbol_names_and_fsms[term_type]

    # TODO FIXME: This is broken, and it's a bug in `lark`'s Python grammar?
    # ptoken = "defx"

    ptoken = "ef foo"
    pmatches = find_partial_matches(term_fsm, ptoken)
    first_pmatch = next(pm for pm in pmatches if pm[0] is not None)
    (parser_state,) = create_pmatch_parser_states(
        lp, terminals_to_states, term_type, ptoken, first_pmatch
    )
    # These copies also patch the lexers in the parse state, which is now
    # needed for use with `parse_to_end`
    parser_state = copy_parser_state(parser_state)
    new_parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert expected_next_tokens == {"NAME"}

    ptoken = "ef foo():"
    pmatches = find_partial_matches(term_fsm, ptoken)
    first_pmatch = next(pm for pm in pmatches if pm[0] is not None)
    (parser_state,) = create_pmatch_parser_states(
        lp, terminals_to_states, term_type, ptoken, first_pmatch
    )
    parser_state = copy_parser_state(parser_state)
    new_parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert not expected_next_tokens

    ptoken = "ef ("
    pmatches = find_partial_matches(term_fsm, ptoken)
    first_pmatch = next(pm for pm in pmatches if pm[0] is not None)
    (parser_state,) = create_pmatch_parser_states(
        lp, terminals_to_states, term_type, ptoken, first_pmatch
    )
    parser_state = copy_parser_state(parser_state)
    with pytest.raises(UnexpectedToken):
        parse_to_end(parser_state)


def test_map_partial_states_to_vocab_regex():
    regex_string = r"([0-9]+([.][0-9]*)?|[.][0-9]+)"
    regex_pattern = interegular.parse_pattern(regex_string)
    regex_fsm = regex_pattern.to_fsm().reduce()

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
    assert possible_paths["FLOAT"] == {0: {1, 2, 3}, 1: {1, 3}, 2: {3}, 3: {3}}

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
