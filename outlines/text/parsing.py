from collections import ChainMap
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import cache
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import interegular
from interegular.fsm import FSM, Alphabet, OblivionError
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.common import LexerConf, ParserConf
from lark.exceptions import (
    LexError,
    UnexpectedCharacters,
    UnexpectedInput,
    UnexpectedToken,
)
from lark.indenter import Indenter
from lark.lexer import (
    BasicLexer,
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
from lark.parsers.lalr_analysis import (
    Action,
    IntParseTable,
    LALR_Analyzer,
    ParseTable,
    Shift,
)
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import LALR_Parser, ParseConf, ParserState, _Parser

PartialParseState = Tuple[str, int]
ParseStateType = Union[int, FrozenSet]


@dataclass
class PartialTerminalInfo:
    priority: int
    terminal_name: str
    can_transition: bool
    is_final: bool


@dataclass
class PartialTokensInfo:
    fsm_state_seq: Tuple[int, ...]
    is_not_finished: bool
    terminals_and_info: Tuple[PartialTerminalInfo, ...]
    final_terminals_and_info: Tuple[PartialTerminalInfo, ...]


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

    def parse_from_state(self, parse_state: "PartialParseState", is_end=False):
        return self.parser.parser.parser.parse_from_state(parse_state, is_end=is_end)


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

        self._termset_fsm_info = None
        self._symbols_to_states: Optional[
            Dict[str, Set[Tuple[ParseStateType, Action]]]
        ] = None
        self._reverse_shifts: Optional[
            Dict[ParseStateType, Dict[str, Set[ParseStateType]]]
        ] = None
        # self._state_transition_map: Optional[
        #     Dict[Tuple[ParseStateType, str], Set[ParseStateType]]
        # ] = None

    def _compute_maps(
        self,
    ):
        """Compute state transition and symbols-to-states maps."""
        self._reverse_shifts = {}
        self._symbols_to_states = {}

        parse_table = self.parser.parser.parse_table

        for from_state, symbols_to_ops in parse_table.states.items():
            for symbol, op in symbols_to_ops.items():
                if op[0] == Shift:
                    symbols_to_from_states = self._reverse_shifts.setdefault(op[1], {})
                    symbols_to_from_states.setdefault(symbol, set()).add(from_state)
                self._symbols_to_states.setdefault(symbol, set()).add((from_state, op))

        # # TODO: This approach is very wasteful.
        # context_lexer = get_contextual_lexer(self)
        # self._state_transition_map = {}
        #
        # for from_state, transitions in parse_table.states.items():
        #     for symbol, action in transitions.items():
        #         # TODO: Filter non-terminals
        #         if symbol not in context_lexer.root_lexer.terminals_by_name:
        #             continue
        #
        #         if action[0] is Shift:
        #             self._state_transition_map.setdefault(
        #                 (from_state, symbol), set()
        #             ).add(action[1])
        #             continue
        #
        #         antecedent_state_seqs = parse_to_terminal(self, [(from_state,)], symbol)
        #
        #         for antecedent_state_seq in antecedent_state_seqs:
        #             antecedent_state = antecedent_state_seq[-1]
        #             self._state_transition_map.setdefault(
        #                 (from_state, symbol), set()
        #             ).add(antecedent_state)

    def _compute_termset_fsm_info(self):
        """Collect and return information about terminal symbol sets and their FSMs.

        Terminal symbol sets (or "termsets") are ordered sequences of terminal
        symbols that are used by each parser state.  Associated with each is a
        collection of FSMs for each terminal and a single parse state FSM that is
        the union of each terminal's FSM.

        This constructs a list of tuples containing the termset, the set of
        parse states that use the termsets, parse state FSMs, and information
        mapping the components of the parse state FSMs to their terminal symbol
        FSMs.

        """
        context_lexer = get_contextual_lexer(self)
        termsets_to_fsms = {}
        termsets_to_parse_states: Dict[Tuple[str, ...], Set[ParseStateType]] = {}
        for parse_state, lexer in context_lexer.lexers.items():
            scanner = lexer.scanner
            key = tuple(term.name for term in scanner.terminals)
            termsets_to_fsms[key] = (scanner.fsm, scanner.fsms_to_trans_finals)
            termsets_to_parse_states.setdefault(key, set()).add(parse_state)

        self._termset_fsm_info = [
            (
                termset,
                frozenset(termsets_to_parse_states[termset]),
                fsm,
                fsms_to_trans_finals,
            )
            for termset, (fsm, fsms_to_trans_finals) in termsets_to_fsms.items()
        ]

    @property
    def termset_fsm_info(self):
        if self._termset_fsm_info is None:
            self._compute_termset_fsm_info()
        return self._termset_fsm_info

    @property
    def symbols_to_states(self):
        if self._symbols_to_states is None:
            self._compute_maps()
        return self._symbols_to_states

    @property
    def reverse_shifts(self):
        if self._reverse_shifts is None:
            self._compute_maps()
        return self._reverse_shifts

    # @property
    # def state_transition_map(self):
    #     if self._state_transition_map is None:
    #         self._compute_maps()
    #     return self._state_transition_map


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

    def feed_token(self, token, is_end=False):
        if token.type == "partial":
            current_state = self.state_stack[-1]
            current_transitions = self.parse_conf.states[current_state]
            current_lexer = get_contextual_lexer(self.lexer).lexers[current_state]

            if not any(
                terminal_info.terminal_name in current_transitions
                or terminal_info.terminal_name in current_lexer.ignore_types
                for terminal_info in token.value.terminals_and_info
            ):
                # If none of the terminals can transition, we should
                # know sooner than later
                expected = {
                    s
                    for s in self.parse_conf.states[current_state].keys()
                    if s.isupper()
                }
                raise UnexpectedToken(
                    token, expected, state=self, interactive_parser=None
                )
            return

        super().feed_token(token, is_end=is_end)


class PartialParser(_Parser):
    def parse(
        self, lexer, start, value_stack=None, state_stack=None, start_interactive=False
    ):
        parse_conf = ParseConf(self.parse_table, self.callbacks, start)
        parser_state = PartialParserState(
            parse_conf, copy(lexer), state_stack, value_stack
        )
        if start_interactive:
            return InteractiveParser(self, parser_state, parser_state.lexer)
        return self.parse_from_state(parser_state)

    def parse_from_state(self, state, last_token=None, is_end=False):
        try:
            token = last_token
            for token in state.lexer.lex(state):
                state.feed_token(token)

            if is_end and (not token or token.type != "partial"):
                end_token = (
                    Token.new_borrow_pos("$END", "", token)
                    if token
                    else Token("$END", "", 0, 1, 1)
                )
                state.feed_token(end_token, True)

            return state
        except UnexpectedInput as e:
            try:
                e.interactive_parser = InteractiveParser(self, state, state.lexer)
            except NameError:
                pass
            raise e
        except Exception:
            if self.debug:
                print("")
                print("STATE STACK DUMP")
                print("----------------")
                for i, s in enumerate(state.state_stack):
                    print("%d)" % i, s)
                print("")

            raise


class PartialScanner(Scanner):
    @classmethod
    @cache
    def construct_terminal_fsm(cls, terminal):
        # TODO: This should really be done at the lexer/parser level so that
        # the lifetime of these objects is tied to the parser itself.
        regex_str = terminal.pattern.to_regexp()
        pattern = interegular.parse_pattern(regex_str)
        fsm, _ = make_deterministic_fsm(pattern.to_fsm().reduce())
        return fsm, pattern.prefix_postfix

    def __init__(self, terminals, g_regex_flags, re_, use_bytes, match_whole=False):
        self.terminals = terminals
        self.g_regex_flags = g_regex_flags
        self.use_bytes = use_bytes
        self.match_whole = match_whole
        self.allowed_types = {t.name for t in self.terminals}
        self._mres = None

        fsms = []
        for t in self.terminals:
            fsm, prefix_postfix = self.construct_terminal_fsm(t)

            # TODO FIXME: We don't support this right now.
            assert prefix_postfix == (0, 0)

            fsms.append(fsm)

        self.fsm, self.fsms_to_trans_finals = fsm_union(fsms)

    def get_terminals_info(
        self, fsm_state_seq
    ) -> Tuple[Tuple[PartialTerminalInfo, ...], Tuple[PartialTerminalInfo, ...]]:
        """Get the possible terminal symbols for an FSM state sequence."""
        terminals_and_info: Tuple[PartialTerminalInfo, ...] = ()
        final_terminals_and_info: Tuple[PartialTerminalInfo, ...] = ()
        for i, (fsm_id, fsm_reads_more, in_final) in enumerate(
            get_sub_fsms_from_seq(fsm_state_seq, self.fsms_to_trans_finals)
        ):
            terminal_name = self.terminals[fsm_id].name
            info = PartialTerminalInfo(i, terminal_name, fsm_reads_more, in_final)
            terminals_and_info += (info,)
            if in_final:
                final_terminals_and_info += (info,)

        return terminals_and_info, final_terminals_and_info

    def match(self, text, pos, last_fsm_state_seq: Optional[Tuple[int, ...]] = None):
        """Determine an FSM match over `text` starting at `pos` and continuing `last_fsm_state_seq`."""

        start_pos = pos

        if last_fsm_state_seq:
            assert len(last_fsm_state_seq) > 1
            start_pos += len(last_fsm_state_seq) - 1
            start_state = last_fsm_state_seq[-1]
        else:
            start_state = self.fsm.initial

        text_part = text[start_pos:]

        res = find_partial_matches(
            self.fsm,
            text_part,
            start_state=start_state,
            full_match=self.match_whole,
        )

        if len(res) == 0:
            return None

        ((_, state_seq),) = res

        if last_fsm_state_seq:
            state_seq = last_fsm_state_seq[:-1] + state_seq

        return state_seq


class PartialContextualLexer(ContextualLexer):
    def __init__(self, conf: "LexerConf", states, always_accept=()):
        terminals = list(conf.terminals)
        terminals_by_name = conf.terminals_by_name

        trad_conf = copy(conf)
        trad_conf.terminals = terminals

        lexer_by_symbols: Dict = {}
        self.lexers = {}
        for state, accepts in states.items():
            key = frozenset(accepts)
            try:
                lexer = lexer_by_symbols[key]
            except KeyError:
                accepts = set(accepts) | set(conf.ignore) | set(always_accept)
                lexer_conf = copy(trad_conf)
                lexer_conf.terminals = [
                    terminals_by_name[n] for n in accepts if n in terminals_by_name
                ]
                lexer = PartialBasicLexer(lexer_conf)
                lexer_by_symbols[key] = lexer

            self.lexers[state] = lexer

        assert trad_conf.terminals is terminals
        self.root_lexer = PartialBasicLexer(trad_conf)

    def lex(self, lexer_state: LexerState, parser_state: Any) -> Iterator[Token]:
        try:
            while True:
                lexer = self.lexers[parser_state.position]
                yield lexer.next_token(lexer_state, parser_state)
        except EOFError:
            pass


class PartialBasicLexer(BasicLexer):
    def __init__(self, conf: "LexerConf"):
        super().__init__(conf)

    def _build_scanner(self):
        # This seems incredibly convoluted: `lark` creates callback-triggered
        # nested scanners for regex-defined terminals that overlap with
        # string-defined terminals when both types of terminals have the same
        # priority.  Unless I'm missing something important, why not simply
        # reorder the terminals so that the string-defined ones come before the
        # regex-defined ones?
        terminals, self.callback = _create_unless(
            self.terminals, self.g_regex_flags, self.re, self.use_bytes
        )

        # We can't let people arbitrarily mess with the scanning process.
        assert not self.user_callbacks
        # for type_, f in self.user_callbacks.items():
        #     if type_ in self.callback:
        #         # Already a callback there, probably UnlessCallback
        #         self.callback[type_] = CallChain(
        #             self.callback[type_], f, lambda t: t.type == type_
        #         )
        #     else:
        #         self.callback[type_] = f

        # We used the "callback" results to reorder the terminals (see the
        # comments above).
        for terminal_name, callback in self.callback.items():
            terminal = self.terminals_by_name[terminal_name]
            for sub_terminal in callback.scanner.terminals:
                self.terminals.remove(sub_terminal)
                idx = self.terminals.index(terminal)
                self.terminals.insert(idx, sub_terminal)

        self._scanner = PartialScanner(
            self.terminals, self.g_regex_flags, self.re, self.use_bytes
        )

    def match(self, text, pos, last_fsm_state_seq=None):
        return self.scanner.match(text, pos, last_fsm_state_seq)

    def next_token(self, lex_state: LexerState, parser_state: Any = None) -> Token:
        last_token = lex_state.last_token

        last_fsm_state_seq = None
        if last_token and last_token.type == "partial":
            # Continue from last partial lexer state
            last_fsm_state_seq = last_token.value.fsm_state_seq

        line_ctr = lex_state.line_ctr
        end_pos = line_ctr.char_pos + (
            len(last_fsm_state_seq) - 1 if last_fsm_state_seq else 0
        )
        while end_pos < len(lex_state.text):
            res = self.match(lex_state.text, line_ctr.char_pos, last_fsm_state_seq)

            if not res:
                if (
                    not last_fsm_state_seq
                    or last_fsm_state_seq[-1] not in self.scanner.fsm.finals
                ):
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

                # The partial match might be complete now
                fsm_state_seq = last_token.value.fsm_state_seq
                terminals_and_info = last_token.value.terminals_and_info
                final_terminals_and_info = last_token.value.final_terminals_and_info
            else:
                fsm_state_seq = res
                (
                    terminals_and_info,
                    final_terminals_and_info,
                ) = self.scanner.get_terminals_info(fsm_state_seq)

            priority_terminal_info = (
                final_terminals_and_info[0]
                if final_terminals_and_info
                else terminals_and_info[0]
            )

            is_not_finished = (
                not priority_terminal_info.is_final
                or priority_terminal_info.can_transition
                or len(terminals_and_info) > 1
            )

            start_pos = line_ctr.char_pos
            end_pos = start_pos + len(fsm_state_seq) - 1

            if end_pos >= len(lex_state.text) and is_not_finished:
                type_name = "partial"
                token_value = PartialTokensInfo(
                    fsm_state_seq,
                    is_not_finished,
                    terminals_and_info,
                    final_terminals_and_info,
                )
                # Don't update the line counter states until we've finished
                value = ""
            else:
                type_name = priority_terminal_info.terminal_name
                # The token value should contain all partial scan parts in this
                # case
                value = token_value = lex_state.text[start_pos:end_pos]

            assert isinstance(self.callback, Dict)

            if type_name not in self.ignore_types:
                t = Token(
                    type_name,
                    token_value,
                    line_ctr.char_pos,
                    line_ctr.line,
                    line_ctr.column,
                )

                line_ctr.feed(value, type_name in self.newline_types)

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

            if type_name in self.callback:
                t2 = Token(
                    type_name, value, line_ctr.char_pos, line_ctr.line, line_ctr.column
                )
                self.callback[type_name](t2)

            line_ctr.feed(value, type_name in self.newline_types)

            last_fsm_state_seq = None

        raise EOFError(self)


class PartialIndenter(Indenter):
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

        # TODO: What do we want to do here?
        # while len(self.indent_level) > 1:
        #     self.indent_level.pop()
        #     yield Token(self.DEDENT_type, "")

    def accepts_token_type(self, token_type):
        if token_type in self.CLOSE_PAREN_types and self.paren_level - 1 < 0:
            return False

        # TODO:
        # if token_type == self.NL_type and self.paren_level == 0:
        #     ...
        #     return False

        return True

    def __copy__(self):
        res = type(self)()
        res.paren_level = self.paren_level
        res.indent_level = copy(self.indent_level)
        return res

    def __repr__(self):
        return f"{type(self).__name__}(paren_level={self.paren_level!r}, indent_level={self.indent_level!r})"


class PartialPythonIndenter(PartialIndenter):
    NL_type = "_NEWLINE"
    OPEN_PAREN_types = ["LPAR", "LSQB", "LBRACE"]
    CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 8


def get_contextual_lexer(x: Union[PartialLexerThread, PartialParsingFrontend]):
    if isinstance(x.lexer, ContextualLexer):
        return x.lexer
    else:
        return x.lexer.lexer


def find_partial_matches(
    fsm: FSM, input_string: str, start_state: Optional[int] = None, full_match=True
) -> Set[Tuple[int, Tuple[int, ...]]]:
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
    full_match
        Matches must cover the entire string.

    Returns
    -------
    A set of tuples corresponding to each valid starting state in the FSM.  The
    first element of each tuple contains an integer indicating the position in
    `input_string` at which the FSM stopped.  The second element is the tuple
    of states visited during execution of the FSM plus the next, unvisited
    transition state.

    """
    if len(input_string) == 0:
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
        last_final_idx = -1

        for i, symbol in enumerate(input_string):
            trans_key = fsm.alphabet[symbol]

            trans_map = fsm_map.get(state)

            if trans_map is None or trans_key not in trans_map:
                if full_match:
                    if state in fsm.finals:
                        i -= 1
                        break
                else:
                    if last_final_idx > -1:
                        i = last_final_idx
                        accepted_states = accepted_states[: last_final_idx + 1]
                        break

                return None, None

            state = trans_map[trans_key]

            if state in fsm.finals:
                last_final_idx = i

            accepted_states += (state,)

        terminated = state in fsm.finals
        if not terminated and state == fsm.initial:
            return None, None

        return i, accepted_states

    res = set()
    transition_maps = (
        fsm.map if start_state is None else {start_state: fsm.map[start_state]}
    )
    for state, trans in transition_maps.items():
        if trans_key in trans:
            last_match_idx, path = _partial_match(trans)
            if last_match_idx is not None and path is not None:
                res.add((last_match_idx, (state,) + path))

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


def fsm_union(
    fsms: Sequence[FSM],
) -> Tuple[FSM, Dict[int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]]]:
    """Construct an FSM representing the union of the FSMs in `fsms`.

    This is an updated version of `interegular.fsm.FSM.union` made to return an
    extra map of component FSMs to the sets of state transitions that
    correspond to them in the new FSM.

    """

    alphabet, new_to_old = Alphabet.union(*[fsm.alphabet for fsm in fsms])

    indexed_fsms = tuple(enumerate(fsms))

    initial = {i: fsm.initial for (i, fsm) in indexed_fsms}

    # Dedicated function accepting a "superset" and returning the next
    # "superset" obtained by following this transition in the new FSM
    def follow(current_state, new_transition: int):
        next = {}
        for i, f in indexed_fsms:
            old_transition = new_to_old[i][new_transition]
            if (
                i in current_state
                and current_state[i] in f.map
                and old_transition in f.map[current_state[i]]
            ):
                next[i] = f.map[current_state[i]][old_transition]
        if not next:
            raise OblivionError
        return next

    states = [initial]
    finals: Set[int] = set()
    map: Dict[int, Dict[int, int]] = {}

    # Map component FSMs to their new state-to-state transitions, finals, and a
    # map translating component FSM states to aggregate FSM states
    fsms_to_trans_finals: Dict[
        int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]
    ] = {}

    i = 0
    while i < len(states):
        state = states[i]

        # Add to the finals of the aggregate FSM whenever we hit a final in a
        # component FSM
        if any(state.get(j, -1) in fsm.finals for (j, fsm) in indexed_fsms):
            finals.add(i)

        # Compute the map for this state
        map[i] = {}
        for transition in alphabet.by_transition:
            try:
                next = follow(state, transition)
            except OblivionError:
                # Reached an oblivion state; don't list it
                continue
            else:
                try:
                    # TODO: Seems like this could--and should--be avoided
                    j = states.index(next)
                except ValueError:
                    j = len(states)
                    states.append(next)

                map[i][transition] = j

                for fsm_id, fsm_state in next.items():
                    (
                        fsm_transitions,
                        fsm_finals,
                        fsm_old_to_new,
                    ) = fsms_to_trans_finals.setdefault(fsm_id, (set(), set(), {}))
                    old_from = state[fsm_id]
                    old_to = fsm_state
                    fsm_old_to_new.setdefault(old_from, set()).add(i)
                    fsm_old_to_new.setdefault(old_to, set()).add(j)
                    fsm_transitions.add((i, j))
                    if fsm_state in fsms[fsm_id].finals:
                        fsm_finals.add(j)

        i += 1

    fsm = FSM(
        alphabet=alphabet,
        states=range(len(states)),
        initial=0,
        finals=finals,
        map=map,
        __no_validation__=True,
    )

    fsm, old_to_new_states = make_deterministic_fsm(fsm)
    _fsms_to_trans_finals = {
        fsm_id: (
            {(old_to_new_states[s1], old_to_new_states[s2]) for s1, s2 in transitions},
            {old_to_new_states[s] for s in finals},
            {
                old_state: {old_to_new_states[new_state] for new_state in new_states}
                for old_state, new_states in old_to_new.items()
            },
        )
        for fsm_id, (transitions, finals, old_to_new) in sorted(
            fsms_to_trans_finals.items(), key=lambda x: x[0]
        )
    }

    return (
        fsm,
        _fsms_to_trans_finals,
    )


def get_sub_fsms_from_seq(
    state_seq: Sequence[int],
    fsms_to_trans_finals: Dict[
        int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]
    ],
) -> Generator[Tuple[int, bool, bool], None, None]:
    """Get the indices of the sub-FSMs in `fsm` that could have matched the state sequence `state_seq`.

    Parameters
    ----------
    state_seq
        A state sequence.
    fsms_to_trans_finals
        A map from FSM indices to tuples containing sets of their state transitions
        and sets of the final/accept states.

    Returns
    -------
    A generator returning tuples containing each sub-FSM index (in the order
    they were union-ed to construct `fsm`) and booleans indicating whether or
    not there is another valid transition from the last state in the sequence
    for the associated sub-FSM (i.e. if the FSM can continue
    accepting/matching) and whether or not the sequence ends in a final state
    of the sub-FSM.
    """
    state_seq_transitions = set(zip(state_seq[:-1], state_seq[1:]))
    last_fsm_state = state_seq[-1]
    yield from (
        (
            # The sub-FMS index
            fsm_idx,
            # Is there another possible transition in this sub-FSM?
            any(last_fsm_state == from_s for (from_s, to_s) in transitions),
            # Is this sub-FSM in a final state?
            state_seq[-1] in finals,
        )
        for fsm_idx, (transitions, finals, _) in fsms_to_trans_finals.items()
        if state_seq_transitions.issubset(transitions)
    )
