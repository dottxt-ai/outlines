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
    parser_state, expected_next_tokens = parse_to_end(ip.parser_state)
    assert not parser_state.value_stack
    assert expected_next_tokens == {"NAME"}

    ip = pyparser.parse_interactive("x = '")
    parser_state, expected_next_tokens = parse_to_end(ip.parser_state)
    assert parser_state.value_stack[-1].type == "EQUAL"
    assert expected_next_tokens == {"LONG_STRING", "STRING"}

    ip = pyparser.parse_interactive("x = 'hi")
    parser_state, expected_next_tokens = parse_to_end(ip.parser_state)
    assert parser_state.value_stack[-1].type == "EQUAL"
    assert expected_next_tokens == {"STRING"}

    ip = pyparser.parse_interactive("x = ('hi")
    parser_state, expected_next_tokens = parse_to_end(ip.parser_state)
    assert parser_state.value_stack[-1].type == "LPAR"
    assert expected_next_tokens == {"STRING"}

    ip = pyparser.parse_interactive("def")
    parser_state, expected_next_tokens = parse_to_end(ip.parser_state)
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
    parser_state = ip.parser_state

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
    name_fsm = name_pattern.to_fsm()

    def_pattern = interegular.parse_pattern("def")
    def_fsm = def_pattern.to_fsm()

    assert find_partial_matches(def_fsm, "def") == {(2, (1, 2, 3))}
    assert find_partial_matches(def_fsm, "de") == {(None, (1, 2))}
    assert find_partial_matches(def_fsm, "d") == {(None, (1,))}
    assert find_partial_matches(def_fsm, "") == set()
    assert find_partial_matches(def_fsm, "df") == set()
    assert find_partial_matches(def_fsm, "ef") == {(1, (2, 3))}
    assert find_partial_matches(def_fsm, "e") == {(None, (2,))}
    assert find_partial_matches(def_fsm, "f") == {(0, (3,))}
    assert find_partial_matches(def_fsm, "ef foo") == {(1, (2, 3))}

    # This string has a `DEF` token in it, but should ultimately not lex one
    assert find_partial_matches(def_fsm, "defb") == {(2, (1, 2, 3))}

    # `NAME` can have multiple start states for this input
    assert find_partial_matches(name_fsm, "d") == {(0, (1,)), (0, (2,))}
    # Not this case
    assert find_partial_matches(name_fsm, "1d") == {(1, (2, 2))}

    assert find_partial_matches(name_fsm, "blah") == {
        (3, (1, 2, 2, 2)),
        (3, (2, 2, 2, 2)),
    }


def test_partial_match_preprocessing():
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

    vocabulary = {"d", "e", "ef foo", "f ", " "}

    pstate_to_vocab = map_partial_states_to_vocab(
        vocabulary, symbol_names_and_fsms, False
    )

    assert dict(pstate_to_vocab) == {
        ("NAME", 1): {"d", "e", "ef foo", "f "},
        ("NAME", 2): {"d", "e", "ef foo", "f "},
        ("DEF", 1): {
            "d",
        },
        ("DEF", 2): {"e", "ef foo"},
        ("DEF", 3): {
            "f ",
        },
        ("__IGNORE_0", 1): {
            " ",
        },
        ("__IGNORE_0", 2): {
            " ",
        },
    }

    pstate_to_vocab = map_partial_states_to_vocab(
        vocabulary, symbol_names_and_fsms, True
    )

    assert dict(pstate_to_vocab) == {
        ("DEF", 1): {"e", "ef foo"},
        ("DEF", 2): {
            "f ",
        },
        ("DEF", 0): {
            "d",
        },
        ("NAME", 1): {"d", "e", "ef foo", "f "},
        ("NAME", 2): {"d", "e", "ef foo", "f "},
        ("NAME", 0): {"d", "e", "ef foo", "f "},
        ("__IGNORE_0", 1): {
            " ",
        },
        ("__IGNORE_0", 2): {
            " ",
        },
        ("__IGNORE_0", 0): {
            " ",
        },
    }


def test_parse_from_partial_match():
    """Make sure we can continue parsing from an FSM-based partial match."""
    pyparser = Lark(
        r"""
start: funcdef

funcdef: "def" name "(" ")" ":"

%ignore /[\t \f]+/  // WS

!name: NAME | "match" | "case"
NAME: /[^\W\d]\w*/

    """,
        parser="lalr",
        postlex=PartialPythonIndenter(),
    )

    terminals_to_states = terminals_to_lalr_states(pyparser)
    symbol_names_and_fsms = terminals_to_fsms(pyparser)

    term_type = "DEF"
    def_fsm = symbol_names_and_fsms[term_type]

    # TODO FIXME: This is broken, and it's a bug in `lark`'s Python grammar!
    # ptoken = "defx"

    ptoken = "ef foo"
    pmatch = find_partial_matches(def_fsm, ptoken)
    first_pmatch = next(pm for pm in pmatch if pm[0] is not None)
    (parser_state,) = create_pmatch_parser_states(
        pyparser, terminals_to_states, term_type, ptoken, first_pmatch
    )
    new_parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert expected_next_tokens == {"NAME"}

    ptoken = "ef foo():"
    pmatch = find_partial_matches(def_fsm, ptoken)
    first_pmatch = next(pm for pm in pmatch if pm[0] is not None)
    (parser_state,) = create_pmatch_parser_states(
        pyparser, terminals_to_states, term_type, ptoken, first_pmatch
    )
    new_parser_state, expected_next_tokens = parse_to_end(parser_state)
    assert not expected_next_tokens

    ptoken = "ef ("
    pmatch = find_partial_matches(def_fsm, ptoken)
    first_pmatch = next(pm for pm in pmatch if pm[0] is not None)
    (parser_state,) = create_pmatch_parser_states(
        pyparser, terminals_to_states, term_type, ptoken, first_pmatch
    )
    with pytest.raises(UnexpectedToken):
        parse_to_end(parser_state)
