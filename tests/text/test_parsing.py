from lark import Lark
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken

from outlines.text.parsing import PartialPythonIndenter, copy_parser_state, parse_to_end


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
