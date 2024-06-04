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
