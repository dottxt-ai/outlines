from copy import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

import regex
from lark.exceptions import (
    LexError,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedToken,
)
from lark.indenter import PythonIndenter
from lark.lexer import BasicLexer, LexerState, Scanner, Token
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.utils import get_regexp_width

if TYPE_CHECKING:
    from lark.lexer import LexerThread
    from lark.parsers.lalr_parser import ParserState


class PartialTokenEOF(UnexpectedEOF):
    pass


class PartialScanner(Scanner):
    def __init__(self, scanner: Scanner):
        self.terminals = scanner.terminals
        self.g_regex_flags = scanner.g_regex_flags
        self.re_ = regex
        self.use_bytes = scanner.use_bytes
        self.match_whole = scanner.match_whole
        self.allowed_types = scanner.allowed_types
        self._mres = scanner._mres

    def match(self, text, pos) -> Optional[Tuple[str, Optional[str], bool]]:
        for mre in self._mres:
            m = mre.match(text, pos=pos, partial=True)
            if m:  # and ((not m.partial) or m.endpos == len(text)):
                return m.group(0), m.lastgroup, m.partial
        return None


class PartialBasicLexer(BasicLexer):
    def __init__(self, basic_lexer: BasicLexer):
        self.re = regex
        self.newline_types = basic_lexer.newline_types
        self.ignore_types = basic_lexer.ignore_types
        self.terminals = basic_lexer.terminals
        self.user_callbacks = basic_lexer.user_callbacks
        self.g_regex_flags = basic_lexer.g_regex_flags
        self.use_bytes = basic_lexer.use_bytes
        self.terminals_by_name = basic_lexer.terminals_by_name
        self.callback = getattr(basic_lexer, "callback", None)

        if basic_lexer._scanner is not None:
            self._scanner: Optional[PartialScanner] = PartialScanner(
                basic_lexer._scanner
            )
        else:
            self._scanner = None

        # This is used to determine the token type for partial matches
        self.terminal_to_regex = {}
        for name, terminal in self.terminals_by_name.items():
            self.terminal_to_regex[name] = self.re.compile(
                terminal.pattern.to_regexp(), self.g_regex_flags
            )

    def _build_scanner(self):
        super()._build_scanner()
        self._scanner = PartialScanner(self._scanner)

    def partial_matches(self, value, type_):
        partial_matches = set()

        # TODO: It's unfortunate that we have to do this costly search (again).
        # It would be better if we could *not* short-circuit the first time we
        # scan in the call to `self.match`.
        for term_name, term_regex in self.terminal_to_regex.items():
            if term_name == type_:
                # A standard lexed token result could actual indicate a partial
                # match
                regex_min, regex_max = get_regexp_width(term_regex.pattern)
                if regex_min <= len(value) < regex_max:
                    partial_matches.add(term_name)
            else:
                m = term_regex.match(value, partial=True)
                if m:
                    partial_matches.add(term_name)

        return partial_matches

    def next_token(self, lex_state: LexerState, parser_state: Any = None) -> Token:
        line_ctr = lex_state.line_ctr
        while line_ctr.char_pos < len(lex_state.text):
            res = self.match(lex_state.text, line_ctr.char_pos)

            if not res:
                allowed = self.scanner.allowed_types - self.ignore_types
                if not allowed:
                    allowed = {"<END-OF-FILE>"}
                raise UnexpectedCharacters(
                    lex_state.text,
                    line_ctr.char_pos,
                    line_ctr.line,
                    line_ctr.column,
                    allowed=allowed,
                    token_history=lex_state.last_token and [lex_state.last_token],
                    state=parser_state,
                    terminals_by_name=self.terminals_by_name,
                )

            value, type_, partial = res

            # Don't advance the lexing state if we're at the end; there could
            # be ambiguous token types that aren't finished.
            if line_ctr.char_pos + len(value) >= len(lex_state.text):
                partial_matches = self.partial_matches(value, type_)
                if partial_matches or partial:
                    raise PartialTokenEOF(partial_matches)

            assert isinstance(self.callback, Dict)

            if type_ not in self.ignore_types:
                t = Token(
                    type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column
                )
                line_ctr.feed(value, type_ in self.newline_types)
                t.end_line = line_ctr.line
                t.end_column = line_ctr.column
                t.end_pos = line_ctr.char_pos
                if t.type in self.callback:
                    t = self.callback[t.type](t)
                    if not isinstance(t, Token):
                        raise LexError(
                            "Callbacks must return a token (returned %r)" % t
                        )
                lex_state.last_token = t
                return t

            if type_ in self.callback:
                t2 = Token(
                    type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column
                )
                self.callback[type_](t2)

            line_ctr.feed(value, type_ in self.newline_types)

        raise EOFError(self)


class PartialPythonIndenter(PythonIndenter):
    """An `Indenter` that doesn't reset its state every time `process` is called."""

    def process(self, stream):
        return self._process(stream)

    def _process(self, stream):
        for token in stream:
            # These were previously *after* the `yield`, but that makes the
            # state tracking unnecessarily convoluted.
            if token.type in self.OPEN_PAREN_types:
                self.paren_level += 1
            elif token.type in self.CLOSE_PAREN_types:
                self.paren_level -= 1
                if self.paren_level < 0:
                    raise UnexpectedToken(token, [])

            if token.type == self.NL_type:
                yield from self.handle_NL(token)
            else:
                yield token

        # while len(self.indent_level) > 1:
        #     self.indent_level.pop()
        #     yield Token(self.DEDENT_type, "")

    def __copy__(self):
        res = type(self)()
        res.paren_level = self.paren_level
        res.indent_level = copy(self.indent_level)
        return res


def copy_lexer_thread(lexer_thread: "LexerThread") -> "LexerThread":
    res = copy(lexer_thread)
    res.lexer = copy(res.lexer)

    if (
        res.lexer.postlexer
        and isinstance(res.lexer.postlexer, PythonIndenter)
        and not isinstance(res.lexer.postlexer, PartialPythonIndenter)
    ):
        # Patch these methods so that the post lexer keeps its state
        # XXX: This won't really work in generality.
        postlexer = PartialPythonIndenter()
        postlexer.paren_level = res.lexer.postlexer.paren_level
        postlexer.indent_level = res.lexer.postlexer.indent_level
        res.lexer.postlexer = postlexer

    # Patch/replace the lexer objects so that they support partial matches
    lexer = res.lexer.lexer
    if not isinstance(lexer.root_lexer, PartialBasicLexer):
        lexer.root_lexer = PartialBasicLexer(lexer.root_lexer)

        basic_lexers = res.lexer.lexer.lexers
        for idx, lexer in basic_lexers.items():
            basic_lexers[idx] = PartialBasicLexer(lexer)

    res.lexer.postlexer = copy(res.lexer.postlexer)

    return res


def copy_parser_state(parser_state: "ParserState") -> "ParserState":
    res = copy(parser_state)
    res.lexer = copy_lexer_thread(res.lexer)

    return res


def copy_ip(ip: "InteractiveParser") -> "InteractiveParser":
    res = copy(ip)
    res.lexer_thread = copy_lexer_thread(res.lexer_thread)
    return res


def parse_to_end(parser_state: "ParserState") -> Tuple["ParserState", Set[str]]:
    """Continue parsing from the current parse state and return partial next tokens."""

    parser_state = copy_parser_state(parser_state)

    expected_next_tokens: Set[str] = set()
    try:
        for token in parser_state.lexer.lex(parser_state):
            parser_state.feed_token(token)
    except PartialTokenEOF as e:
        expected_next_tokens = e.expected

    return parser_state, expected_next_tokens
