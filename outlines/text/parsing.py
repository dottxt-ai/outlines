from collections import ChainMap, defaultdict
from copy import copy, deepcopy
from typing import Any, Callable, DefaultDict, Dict, Iterable, Optional, Set, Tuple

import interegular
import regex
from interegular.fsm import FSM, Alphabet, anything_else
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.common import LexerConf, ParserConf
from lark.exceptions import (
    LexError,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedToken,
)
from lark.indenter import PythonIndenter
from lark.lexer import (
    BasicLexer,
    CallChain,
    ContextualLexer,
    LexerState,
    LexerThread,
    Scanner,
    _create_unless,
)
from lark.parser_frontends import (
    ParsingFrontend,
    PostLexConnector,
    _validate_frontend_args,
)
from lark.parsers.lalr_analysis import IntParseTable, LALR_Analyzer, ParseTable, Shift
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import LALR_Parser, ParseConf, ParserState, _Parser
from lark.utils import get_regexp_width

PartialParseState = Tuple[str, int]


def make_deterministic_fsm(fsm: FSM) -> Tuple[FSM, Dict[int, int]]:
    """Construct an equivalent FSM with deterministic state labels."""
    old_to_new_trans_keys = {
        trans_key: i
        for i, (trans_key, _) in enumerate(
            sorted(fsm.alphabet.by_transition.items(), key=lambda x: sorted(x[1]))
        )
    }

    new_symbol_mapping = {
        symbol: old_to_new_trans_keys[trans_key]
        for symbol, trans_key in fsm.alphabet._symbol_mapping.items()
    }

    new_alphabet = Alphabet(new_symbol_mapping)

    new_map = {
        from_state: {
            old_to_new_trans_keys[trans_key]: to_state
            for trans_key, to_state in trans_map.items()
        }
        for from_state, trans_map in fsm.map.items()
    }

    old_to_new_states = {}
    old_to_new_states[fsm.initial] = 0

    i = 0
    seen = {fsm.initial}
    old_state_queue = [fsm.initial]
    while old_state_queue:
        old_state = old_state_queue.pop(-1)
        transitions = new_map[old_state]
        sorted_transitions = sorted(transitions.items(), key=lambda v: v[0])
        for _, old_state in sorted_transitions:
            if old_state not in seen:
                old_state_queue.append(old_state)
                seen.add(old_state)
            if old_state not in old_to_new_states:
                i += 1
                old_to_new_states[old_state] = i

    new_map = dict(
        sorted(
            (
                (
                    old_to_new_states[from_state],
                    dict(
                        sorted(
                            (
                                (trans_key, old_to_new_states[to_state])
                                for trans_key, to_state in trans_map.items()
                            ),
                            key=lambda v: v[0],
                        )
                    ),
                )
                for from_state, trans_map in new_map.items()
            ),
            key=lambda v: v[0],
        )
    )

    new_initial = 0
    new_finals = frozenset(
        sorted(old_to_new_states[old_state] for old_state in fsm.finals)
    )
    new_states = frozenset(sorted(new_map.keys()))

    new_fsm = FSM(new_alphabet, new_states, new_initial, new_finals, new_map)

    return new_fsm, old_to_new_states


class PartialTokenEOF(UnexpectedEOF):
    pass


class PartialParserConf(ParserConf):
    __serialize_fields__ = "rules", "start", "parser_type", "deterministic"

    def __init__(self, rules, callbacks, start, deterministic):
        super().__init__(rules, callbacks, start)
        self.deterministic = deterministic


class PartialLark(Lark):
    __serialize_fields__ = "parser", "rules", "options", "deterministic"

    def __init__(self, grammar, **options):
        # TODO: Could've extended `LarkOptions`, but all these extensions are
        # already way too much (and brittle).  This library really needs a
        # complete refactoring.
        self.deterministic = options.pop("deterministic", False)
        options["regex"] = True
        super().__init__(grammar, **options)
        assert self.options.parser == "lalr"

    def _build_lexer(self, dont_ignore: bool = False) -> "PartialBasicLexer":
        lexer_conf = self.lexer_conf
        if dont_ignore:
            from copy import copy

            lexer_conf = copy(lexer_conf)
            lexer_conf.ignore = ()

        return PartialBasicLexer(lexer_conf)

    def _build_parser(self) -> "PartialParsingFrontend":
        self._prepare_callbacks()
        _validate_frontend_args(self.options.parser, self.options.lexer)
        parser_conf = PartialParserConf(
            self.rules, self._callbacks, self.options.start, self.deterministic
        )

        # This is `_construct_parsing_frontend` expanded/inlined
        parser_type = self.options.parser
        lexer_type = self.options.lexer
        lexer_conf = self.lexer_conf

        assert isinstance(lexer_conf, LexerConf)
        assert isinstance(parser_conf, ParserConf)
        parser_conf.parser_type = parser_type
        self.lexer_conf.lexer_type = lexer_type
        return PartialParsingFrontend(lexer_conf, parser_conf, self.options)

    def __repr__(self):
        return "{}(open({!r}), parser={!r}, lexer={!r}, ...)".format(
            type(self).__name__,
            self.source_path,
            self.options.parser,
            self.options.lexer,
        )


class PartialLexerThread(LexerThread):
    def __copy__(self):
        return type(self)(copy(self.lexer), copy(self.state))

    def __repr__(self):
        return f"{type(self).__name__}(lexer={self.lexer!r}, state={self.state!r})"


class PartialPostLexConnector(PostLexConnector):
    def __copy__(self):
        return type(self)(self.lexer, copy(self.postlexer))

    def __repr__(self):
        return (
            f"{type(self).__name__}(lexer={self.lexer!r}, postlexer={self.postlexer!r})"
        )


class PartialParsingFrontend(ParsingFrontend):
    def __init__(self, lexer_conf, parser_conf, options, parser=None):
        assert parser_conf.parser_type == "lalr"

        options._plugins["LALR_Parser"] = PartialLALRParser
        options._plugins["BasicLexer"] = PartialBasicLexer
        options._plugins["ContextualLexer"] = PartialContextualLexer
        options._plugins["LexerThread"] = PartialLexerThread

        super().__init__(lexer_conf, parser_conf, options, parser=parser)

        if lexer_conf.postlex:
            self.lexer = PartialPostLexConnector(self.lexer.lexer, lexer_conf.postlex)


class PartialLALRParser(LALR_Parser):
    def __init__(self, parser_conf, debug=False, strict=False):
        analysis = LALR_Analyzer(
            parser_conf, debug=debug if not parser_conf.deterministic else True
        )
        analysis.compute_lalr()
        callbacks = parser_conf.callbacks

        self.parser_conf = parser_conf
        self._parse_table = analysis.parse_table

        if parser_conf.deterministic:
            old_to_new = {}

            def to_tuple(v):
                new = old_to_new.get(v)
                if new is None:
                    new = tuple(sorted(v, key=lambda y: str(y)))
                    old_to_new[v] = new
                return new

            enum = sorted(
                self._parse_table.states.keys(),
                key=lambda x: str(sorted(x, key=lambda y: str(y))),
            )

            new_states = {}
            for s in enum:
                transitions = {
                    term: op if op[0] is not Shift else (op[0], to_tuple(op[1]))
                    for term, op in self._parse_table.states[s].items()
                }
                new_states[to_tuple(s)] = transitions

            self._parse_table = type(self._parse_table)(
                new_states,
                {k: to_tuple(v) for k, v in self._parse_table.start_states.items()},
                {k: to_tuple(v) for k, v in self._parse_table.end_states.items()},
            )

            if not debug:
                self._parse_table = IntParseTable.from_ParseTable(self._parse_table)
                self.states_to_rulesets = dict(
                    zip(self._parse_table.states.keys(), new_states.keys())
                )

        self.parser = PartialParser(self._parse_table, callbacks, debug)

    @classmethod
    def deserialize(cls, data, memo, callbacks, debug=False):
        inst = cls.__new__(cls)
        inst._parse_table = ParseTable.deserialize(data, memo)
        inst.parser = PartialParser(inst._parse_table, callbacks, debug)
        return inst


class PartialParserState(ParserState):
    def __copy__(self):
        return type(self)(
            self.parse_conf,
            copy(self.lexer),
            copy(self.state_stack),
            deepcopy(self.value_stack),
        )

    def __repr__(self):
        return f"{type(self).__name__}(lexer={self.lexer!r}, state_stack={self.state_stack!r})"


class PartialParser(_Parser):
    def parse(
        self, lexer, start, value_stack=None, state_stack=None, start_interactive=False
    ):
        parse_conf = ParseConf(self.parse_table, self.callbacks, start)
        parser_state = PartialParserState(parse_conf, lexer, state_stack, value_stack)
        if start_interactive:
            return InteractiveParser(self, parser_state, parser_state.lexer)
        return self.parse_from_state(parser_state)


class PartialScanner(Scanner):
    def __init__(self, terminals, g_regex_flags, re_, use_bytes, match_whole=False):
        self.terminals = terminals
        self.g_regex_flags = g_regex_flags
        self.re_ = regex
        self.use_bytes = use_bytes
        self.match_whole = match_whole
        self.allowed_types = {t.name for t in self.terminals}
        self._mres = self._build_mres(terminals, len(terminals))

    def match(self, text, pos) -> Optional[Tuple[str, Optional[str], bool]]:
        for mre in self._mres:
            m = mre.match(text, pos=pos, partial=True)
            if m:  # and ((not m.partial) or m.endpos == len(text)):
                return m.group(0), m.lastgroup, m.partial
        return None


class PartialContextualLexer(ContextualLexer):
    def __init__(self, conf: "LexerConf", states, always_accept=()):
        terminals = list(conf.terminals)
        terminals_by_name = conf.terminals_by_name

        trad_conf = copy(conf)
        trad_conf.terminals = terminals

        lexer_by_tokens: Dict = {}
        self.lexers = {}
        for state, accepts in states.items():
            key = frozenset(accepts)
            try:
                lexer = lexer_by_tokens[key]
            except KeyError:
                accepts = set(accepts) | set(conf.ignore) | set(always_accept)
                lexer_conf = copy(trad_conf)
                lexer_conf.terminals = [
                    terminals_by_name[n] for n in accepts if n in terminals_by_name
                ]
                lexer = PartialBasicLexer(lexer_conf)
                lexer_by_tokens[key] = lexer

            self.lexers[state] = lexer

        assert trad_conf.terminals is terminals
        self.root_lexer = PartialBasicLexer(trad_conf)


class PartialBasicLexer(BasicLexer):
    def __init__(self, conf: "LexerConf"):
        super().__init__(conf)

        # This is used to determine the token type for partial matches
        self.terminal_to_regex = {}
        for name, terminal in self.terminals_by_name.items():
            self.terminal_to_regex[name] = self.re.compile(
                terminal.pattern.to_regexp(), self.g_regex_flags
            )

    def _build_scanner(self):
        terminals, self.callback = _create_unless(
            self.terminals, self.g_regex_flags, self.re, self.use_bytes
        )
        assert all(self.callback.values())

        for type_, f in self.user_callbacks.items():
            if type_ in self.callback:
                # Already a callback there, probably UnlessCallback
                self.callback[type_] = CallChain(
                    self.callback[type_], f, lambda t: t.type == type_
                )
            else:
                self.callback[type_] = f

        self._scanner = PartialScanner(
            terminals, self.g_regex_flags, self.re, self.use_bytes
        )

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

    def __repr__(self):
        return f"{type(self).__name__}(paren_level={self.paren_level!r}, indent_level={self.indent_level!r})"


def parse_to_end(
    parser_state: PartialParserState,
) -> Tuple[PartialParserState, Set[str]]:
    """Continue parsing from the current parse state and return partial next tokens.

    .. warning::
        The parse state `parser_state` is updated in-place and must be patched
        to work with this function.

    """

    expected_next_tokens: Set[str] = set()
    try:
        for token in parser_state.lexer.lex(parser_state):
            parser_state.feed_token(token)
    except PartialTokenEOF as e:
        expected_next_tokens = e.expected

    return parser_state, expected_next_tokens


def find_partial_matches(
    fsm: FSM, input_string: str, start_state: Optional[int] = None
) -> Set[Tuple[Optional[int], Tuple[int, ...]]]:
    """Find the states in the finite state machine `fsm` that accept `input_string`.

    This will consider all possible states in the finite state machine (FSM)
    that accept the beginning of `input_string` as starting points, unless a
    specific `start_state` is provided.

    Parameters
    ----------
    fsm
        The finite state machine.
    input_string
        The string for which we generate partial matches.
    start_state
        A single fixed starting state to consider.  For example, if this value
        is set to `fsm.initial`, it attempt to read `input_string` from the
        beginning of the FSM/regular expression.

    Returns
    -------
    A set of tuples corresponding to each valid starting state in the FSM.  The
    first element of each tuple contains either ``None`` or an integer
    indicating the position in `input_string` at which the FSM terminated.  The
    second element is the tuple of states visited during execution of the FSM
    plus the next, unvisited transition state.

    """
    if len(input_string) == 0 or input_string[0] not in fsm.alphabet:
        return set()

    trans_key = fsm.alphabet[input_string[0]]

    # TODO: We could probably reuse parts of the computed paths when computing
    # results for multiple starting points.
    def _partial_match(
        trans: Dict[int, int]
    ) -> Tuple[Optional[int], Optional[Tuple[int, ...]]]:
        fsm_map = ChainMap({fsm.initial: trans}, fsm.map)
        state = fsm.initial
        accepted_states: Tuple[int, ...] = ()

        for i, symbol in enumerate(input_string):
            if anything_else in fsm.alphabet and symbol not in fsm.alphabet:
                symbol = anything_else

            trans_key = fsm.alphabet[symbol]

            if not (state in fsm_map and trans_key in fsm_map[state]):
                if state in fsm.finals:
                    i -= 1
                    break
                return None, None

            state = fsm_map[state][trans_key]

            accepted_states += (state,)

        terminated = state in fsm.finals
        if not terminated and state == fsm.initial:
            return None, None

        return None if not terminated else i, accepted_states

    res = set()
    transition_maps = (
        fsm.map if start_state is None else {start_state: fsm.map[start_state]}
    )
    for state, trans in transition_maps.items():
        if trans_key in trans:
            n_matched, path = _partial_match(trans)
            if path is not None:
                res.add((n_matched, (state,) + path))

    return res


def terminals_to_fsms(lp: PartialLark) -> Dict[str, FSM]:
    """Construct a ``dict`` mapping terminal symbol names to their finite state machines."""

    symbol_names_and_fsms = {}
    for terminal in lp.terminals:
        pattern = interegular.parse_pattern(terminal.pattern.to_regexp())
        # TODO: Use `pyparser.terminals[0].pattern.flags`?
        try:
            fsm, _ = make_deterministic_fsm(pattern.to_fsm().reduce())
        except Unsupported:
            fsm = None

        symbol_names_and_fsms[terminal.name] = fsm

    return symbol_names_and_fsms


def map_partial_states_to_vocab(
    vocabulary: Iterable[str],
    terminals_to_fsms_map: Dict[str, FSM],
    partial_match_filter: Callable[
        [str, Optional[int], Tuple[int, ...]], bool
    ] = lambda *args: True,
    final_state_string: Optional[str] = None,
) -> Tuple[Dict[PartialParseState, Set[int]], Dict[str, Dict[int, Set[int]]]]:
    """Construct a map from partial parse states to subsets of `vocabulary`.

    The subsets of `vocabulary` consist of elements that are accepted by--or
    transition to--the corresponding partial parse states.

    Parameters
    ----------
    vocabulary
        The vocabulary composed of strings.
    terminals_to_fsms_map
        Terminal symbol names mapped to FSMs, as provided by `terminals_to_fsms`.
    partial_match_filter
        A callable that determines which partial matches to keep.  The first
        argument is the string being match, the rest are the unpacked partial
        match return values of `find_partial_matches`.
    final_state_string
        A string from `vocabulary` that is to be added to all the final states
        in the FSM.
    """

    final_state_string_idx = None

    # Partial parse states to the subsets of the vocabulary that accept them
    pstate_to_vocab: Dict[Tuple[str, int], Set[int]] = {}
    possible_paths = {}
    for symbol_name, fsm in terminals_to_fsms_map.items():
        terminal_possible_paths: Dict[int, Set[int]] = {}
        for i, vocab_string in enumerate(vocabulary):
            if vocab_string == final_state_string:
                final_state_string_idx = i

            for end_idx, state_seq in find_partial_matches(fsm, vocab_string):
                if partial_match_filter(vocab_string, end_idx, state_seq):
                    terminal_possible_paths.setdefault(state_seq[0], set()).add(
                        state_seq[-1]
                    )
                    pstate_to_vocab.setdefault((symbol_name, state_seq[0]), set()).add(
                        i
                    )

        possible_paths[symbol_name] = terminal_possible_paths

    if final_state_string_idx is not None:
        # Allow transitions to EOS from all terminals FSM states
        for symbol_name, fsm in terminals_to_fsms_map.items():
            for state in fsm.finals:
                pstate_to_vocab.setdefault((symbol_name, state), set()).add(
                    final_state_string_idx
                )

    return pstate_to_vocab, possible_paths


def terminals_to_lalr_states(lp: PartialLark) -> DefaultDict[str, Set[int]]:
    terminals_to_states = defaultdict(set)
    parse_table = lp.parser.parser.parser.parse_table
    for state, tokens_to_ops in parse_table.states.items():
        for token, op in tokens_to_ops.items():
            if op[0] == Shift:
                # `op[1]` is the state we shift to when `token` is observed
                terminals_to_states[token].add(op[1])

    return terminals_to_states


def create_pmatch_parser_states(
    lp: PartialLark,
    terminals_to_states: Dict[str, Set[int]],
    term_type: str,
    ptoken: str,
    pmatch: Tuple[int, Tuple[int, ...]],
) -> Tuple[PartialParserState, ...]:
    parse_table = lp.parser.parser.parser.parse_table

    # TODO: We need to effectively disable the callbacks that build the
    # trees, because we aren't actually parsing a valid state that can, say,
    # be reduced
    def noop(*args, **kwargs):
        pass

    callbacks = {rule: noop for rule, cb in lp._callbacks.items()}
    parse_conf = ParseConf(parse_table, callbacks, lp.options.start[0])
    lexer_thread = lp.parser._make_lexer_thread(ptoken)
    lexer_state = lexer_thread.state
    lexer_state.line_ctr.char_pos = pmatch[0] + 1
    lexer_state.last_token = Token(term_type, "")
    res = tuple(
        PartialParserState(parse_conf, lexer_thread, [state], None)
        for state in terminals_to_states[term_type]
    )
    return res
