import collections
import copy
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Union,
    List,
    Optional,
    Callable,
    Tuple,
    Set,
    Dict,
    FrozenSet,
    Sequence,
    cast,
)

import interegular
import torch
from interegular.fsm import (
    Alphabet,
    FSM,
    anything_else,
    _AnythingElseCls,
    TransitionKey,
    State,
)
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken
from outlines_core import Guide as CoreGuide, Index, Vocabulary
from outlines_core.json_schema import build_regex_from_schema
from outlines import grammars
from outlines.fsm.parsing import PartialLark, PartialParserState, make_deterministic_fsm

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


@dataclass(frozen=True)
class Write:
    """Write instruction.

    Attributes
    ----------
    tokens
        The sequence of tokens to be added to the current sequence by the
        generation process.

    """

    tokens: List[int]


@dataclass(frozen=True)
class Generate:
    """Generate instruction

    Attributes
    ----------
    tokens
        The tokens that lead to a valid completion if generated.  A value
        of ``None`` indicates that all tokens are allowed.
    """

    tokens: Optional[List[int]]


Instruction = Union[Write, Generate]


class Guide:
    """Base definition of a generation guide.

    A generation guide defines the behavior of a finite-state machine that guides
    a text generation procedure. Unlike the DFAs built from regular expressions
    guides can also emit a `Write` instructions which tells the model that it can
    append a sequence of tokens (or token word) instead of generating it.

    """

    def __init__(self, index: Index):
        self._core_guide = CoreGuide(index)
        self.initial_state = self._core_guide.get_state()

    def get_next_instruction(self, state: Any) -> Instruction:
        """Retrieve the list of allowed tokens for the given state."""
        return self._core_guide.get_tokens()  # type: ignore

    def get_next_state(self, state: Any, token_id: int) -> Any:
        """Advance the guide's state with a token and return the new state."""
        self._core_guide.advance(token_id)
        return self._core_guide.get_state()

    def is_final_state(self, state: Any) -> bool:
        """Check if the given state is a final state."""
        return self._core_guide.is_finished()

    def copy(self) -> "Guide":
        """Create a copy of the Guide."""
        new_guide = Guide(self._core_guide.index)  # Assuming index is accessible
        return new_guide

    def __repr__(self) -> str:
        return repr(self._core_guide)

    def __str__(self) -> str:
        return str(self._core_guide)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Guide):
            return False
        return self._core_guide == other._core_guide


class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""

    final_state = 1
    start_state = 0  # TODO: remove start_state, use only initial_state
    initial_state = 0

    def __init__(self, tokenizer: "Tokenizer"):
        """Initialize the generation guide.

        model
            The logit generator used to generate the next token.

        """
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary.values()

    def get_next_instruction(self, state: int) -> Instruction:
        if self.is_final_state(state):
            return Write([self.eos_token_id])
        return Generate(None)

    def get_next_state(self, state: int, token_id: int) -> int:
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state

        return self.initial_state

    def is_final_state(self, state: int):
        return state == self.final_state

    def copy(self):
        return self


def uncached_create_states_mapping(
    regex_string: str,
    tokenizer,
    regex_parser: Callable[[str], interegular.Pattern] = interegular.parse_pattern,
    frozen_tokens: List[str] = [],
) -> Tuple[Index, Set[int], Set[int]]:
    """Create the variables related to the mapping between states and tokens from a regex string.

    The parameters of the function are used for caching purpose.

    Parameters
    ----------
    regex_string:
        The regular expression string to generate a states mapping for.
    tokenizer:
        The model's tokenizer.
    regex_parser:
        A function that parses a regex string into an `interegular` Pattern object.
    frozen_tokens:
        A list of tokens that should be kept as-is when expanding the token-level FSM
        into a byte-level FSM. Defaults to an empty list.

    Returns
    -------
    states_to_token_maps:
        A mapping from states to a mapping from token ids originating from that state
        to the next state to transition to given that token. The structure is as follows:
        (origin_state -> (token_id -> next_state))
    empty_token_ids:
        A set of token ids that correspond to empty strings.
    final_states:
        A set of final states in the FSM.
    """
    regex_fsm = regex_parser(regex_string).to_fsm()
    return create_states_mapping_from_fsm(regex_fsm, tokenizer, frozen_tokens)


def cached_create_states_mapping(regex_string, tokenizer, *args, **kwargs):
    return uncached_create_states_mapping(regex_string, tokenizer, *args, **kwargs)


def create_states_mapping(
    regex_string: str,
    tokenizer,
    regex_parser: Callable[[str], interegular.Pattern] = interegular.parse_pattern,
    frozen_tokens: List[str] = [],
) -> Tuple[Index, Set[int], Set[int]]:
    """Create the variables related to the mapping between states and tokens from a regex string.

    The parameters of the function are used for caching purpose.

    Parameters
    ----------
    regex_string:
        The regular expression string to generate a states mapping for.
    tokenizer:
        The model's tokenizer.
    regex_parser:
        A function that parses a regex string into an `interegular` Pattern object.
    frozen_tokens:
        A list of tokens that should be kept as-is when expanding the token-level FSM
        into a byte-level FSM. Defaults to an empty list.

    Returns
    -------
    states_to_token_maps:
        A mapping from states to a mapping from token ids originating from that state
        to the next state to transition to given that token. The structure is as follows:
        (origin_state -> (token_id -> next_state))
    empty_token_ids:
        A set of token ids that correspond to empty strings.
    final_states:
        A set of final states in the FSM.
    """
    regex_fsm = regex_parser(regex_string).to_fsm()
    return create_states_mapping_from_fsm(regex_fsm, tokenizer, frozen_tokens)


TransitionTrie = Dict[TransitionKey, "Union[TransitionTrie, State, None]"]


def add_to_transition_trie(
    trie: TransitionTrie,
    key_seq: Sequence[TransitionKey],
    value: Union[State, None],
):
    for key in key_seq[:-1]:
        trie = cast(TransitionTrie, trie.setdefault(key, {}))
        assert isinstance(trie, dict), "key sequence of incompatible length"
    trie[key_seq[-1]] = value


# merge default_trie into the trie, only updating entries not present in the trie
def transition_trie_setdefault(
    trie: TransitionTrie,
    default_trie: TransitionTrie,
):
    for key, default_value in default_trie.items():
        dest_value = trie.get(key)
        if isinstance(dest_value, dict) and isinstance(default_value, dict):
            transition_trie_setdefault(dest_value, default_value)
        elif key not in trie:
            trie[key] = default_value


def byte_symbol(byte: int) -> str:
    return f"\x00{byte:02X}" if byte >= 0x80 else chr(byte)


def make_byte_level_fsm(
    fsm: FSM, keep_utf8: bool = False, frozen_tokens: List[str] = []
) -> FSM:
    """Convert an FSM to a byte-level FSM, expanding multibyte characters as
    sequences of single-byte transitions.

    Parameters
    ----------
    fsm: (`interegular.FSM`):
        The token-level FSM to convert to a byte-level FSM.
    keep_utf8: (`bool`, *optional*):
        If set to True, the original utf-8 characters are kept as-is. Defaults to
        False. NOTE: we're representing bytes as strings to keep it type-compatible.
    frozen_tokens: (`List[str]`, *optional*):
        A list of tokens that should be kept as-is in the byte-level FSM. That is,
        these tokens will not be expanded into byte-level transitions. Defaults to
        an empty list.

    Returns
    -------
    `interegular.FSM`: A byte-level FSM.
    """

    anything_else_key = fsm.alphabet[anything_else]
    symbol_mapping: Dict[Union[str, _AnythingElseCls], TransitionKey] = {}
    map: Dict[State, Dict[TransitionKey, State]] = {}
    states: List[State] = list(fsm.states)

    # identify all multibyte characters in the alphabet and build a mapping
    # from the original transition keys to sequences of new keys for each byte
    key_to_key_seqs: Dict[TransitionKey, Set[Tuple[TransitionKey, ...]]] = {}
    all_key_seqs: Set[Tuple[TransitionKey, ...]] = set()
    all_bytes: Set[int] = set()
    max_key = max(fsm.alphabet.values())
    for symbol, transition_key in fsm.alphabet.items():
        assert symbol == anything_else or symbol in frozen_tokens or len(symbol) == 1
        if symbol == anything_else or symbol in frozen_tokens or ord(symbol) < 0x80:
            symbol_mapping[symbol] = transition_key
        else:
            if keep_utf8:
                symbol_mapping[symbol] = transition_key
            key_list: List[TransitionKey] = []
            for byte in symbol.encode("utf-8"):
                symbol = byte_symbol(byte)
                if symbol not in symbol_mapping:
                    symbol_mapping[symbol] = max_key = TransitionKey(max_key + 1)
                    all_bytes.add(byte)
                key_list.append(symbol_mapping[symbol])
            key_seq = tuple(key_list)
            key_to_key_seqs.setdefault(transition_key, set()).add(key_seq)
            all_key_seqs.add(key_seq)

    # add all remaining multi-byte utf-8 bytes to the alphabet
    # (this is required to represent `anything_else`)
    utf8_ranges = {
        1: (0x80, 0xC0),  # continuation bytes
        2: (0xC0, 0xE0),  # 2-byte sequences
        3: (0xE0, 0xF0),  # 3-byte sequences
        4: (0xF0, 0xF8),  # 4-byte sequences
    }
    utf8_all_keys: Dict[int, Set[TransitionKey]] = {
        n: set() for n in utf8_ranges.keys()
    }
    for n, (start, end) in utf8_ranges.items():
        range_key = max_key = TransitionKey(max_key + 1)
        for byte in range(start, end):
            byte_key = symbol_mapping.setdefault(byte_symbol(byte), range_key)
            utf8_all_keys[n].add(byte_key)

    # cache of intermediate transition states by transitions from that state
    state_cache: Dict[FrozenSet[Tuple[TransitionKey, State]], State] = {}

    # helper function to create multi-step transitions between states
    max_state = max(fsm.states)

    def create_seq_transitions(
        seq_transitions_trie: TransitionTrie,
    ) -> Dict[TransitionKey, State]:
        nonlocal max_state
        result: Dict[TransitionKey, State] = {}

        for next_key, next_trie in seq_transitions_trie.items():
            if isinstance(next_trie, dict):
                next_transitions = create_seq_transitions(next_trie)
                if not next_transitions:
                    continue
                cache_key = frozenset(next_transitions.items())
                next_state = state_cache.get(cache_key)
                if next_state is None:
                    next_state = max_state = State(max_state + 1)
                    map[next_state] = next_transitions
                    state_cache[cache_key] = next_state
                    states.append(next_state)
                result[next_key] = next_state
            elif next_trie is not None:
                result[next_key] = next_trie

        return result

    # create new states and transitions
    for state, transitions in fsm.map.items():
        seq_transitions_trie: TransitionTrie = {}
        state_map: Dict[TransitionKey, State] = {}

        for transition_key, to_state in transitions.items():
            if transition_key in key_to_key_seqs:
                if keep_utf8:
                    state_map[transition_key] = to_state
                for key_seq in key_to_key_seqs[transition_key]:
                    add_to_transition_trie(seq_transitions_trie, key_seq, to_state)
            else:  # keep single-byte transitions as is
                state_map[transition_key] = to_state

        # handle multi-byte anything_else sequences
        if anything_else_key in transitions:
            for key_seq in all_key_seqs:
                add_to_transition_trie(seq_transitions_trie, key_seq, None)

            anything_else_trie: TransitionTrie = {}
            cont_trie: Union[TransitionTrie, State] = transitions[anything_else_key]
            for n in range(2, 5):
                cont_trie = {key: cont_trie for key in utf8_all_keys[1]}
                for key in utf8_all_keys[n]:
                    anything_else_trie[key] = cont_trie

            transition_trie_setdefault(seq_transitions_trie, anything_else_trie)

        # create new states and transitions
        next_transitions = create_seq_transitions(seq_transitions_trie)
        state_map.update(next_transitions)
        map[state] = state_map

    return FSM(
        alphabet=Alphabet(symbol_mapping),
        states=states,
        initial=fsm.initial,
        finals=fsm.finals,
        map=map,
    )


def create_fsm_index_tokenizer(regex_fsm, tokenizer):
    pass  # FIXME: was implement as Rust code in outlines_core==0.1.26


def create_states_mapping_from_fsm(
    fsm: interegular.fsm.FSM,
    tokenizer,
    frozen_tokens: List[str] = [],
) -> Tuple[Index, Set[int], Set[int]]:
    """Create the variables related to the mapping between states and tokens from an FSM.

    The parameters of the function are used for caching purpose.

    Parameters
    ----------
    fsm:
        An FSM for the regular expression.
    tokenizer:
        The model's tokenizer.
    frozen_tokens:
        A list of tokens that should be kept as-is when expanding the token-level FSM
        into a byte-level FSM. Defaults to an empty list.

    Returns
    -------
    states_to_token_maps:
        A mapping from states to a mapping from token ids originating from that state
        to the next state to transition to given that token. The structure is as follows:
        (origin_state -> (token_id -> next_state))
    empty_token_ids:
        A set of token ids that correspond to empty strings.
    final_states:
        A set of final states in the FSM.
    """
    byte_fsm = make_byte_level_fsm(
        fsm.reduce(), keep_utf8=True, frozen_tokens=frozen_tokens
    )
    regex_fsm, _ = make_deterministic_fsm(byte_fsm)
    states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
        regex_fsm, tokenizer
    )

    return states_to_token_maps, empty_token_ids, regex_fsm.finals


class CoreRegexGuide(Guide):
    """Guide to generate text in the language of a regular expression."""

    initial_state = 0

    def __init__(
        self,
        states_to_token_maps,
        empty_token_ids,
        fsm_finals,
        eos_token_id,
        states_to_token_mask,
    ):
        self.states_to_token_maps = states_to_token_maps
        self.empty_token_ids = empty_token_ids
        self.eos_token_id = eos_token_id
        self.final_states = fsm_finals | {-1}
        self.states_to_token_mask = states_to_token_mask

    @classmethod
    def from_regex(
        cls,
        regex_string: str,
        tokenizer,
        _create_states_mapping=create_states_mapping,
        device=None,
        regex_parser: Callable[[str], interegular.Pattern] = interegular.parse_pattern,
        frozen_tokens: List[str] = [],
    ):
        (
            states_to_token_maps,
            empty_token_ids,
            fsm_finals,
        ) = _create_states_mapping(
            regex_string,
            tokenizer,
            regex_parser=regex_parser,
            frozen_tokens=frozen_tokens,
        )
        states_to_token_mask = {
            state: torch.tensor(list(next_tokens_to_end_states.keys()), device=device)
            for state, next_tokens_to_end_states in states_to_token_maps.items()
        }
        return cls(
            states_to_token_maps,
            empty_token_ids,
            fsm_finals,
            tokenizer.eos_token_id,
            states_to_token_mask,
        )

    @classmethod
    def from_interegular_fsm(
        cls,
        interegular_fsm: interegular.fsm.FSM,
        tokenizer,
        _create_states_mapping_from_fsm=create_states_mapping_from_fsm,
        device=None,
        frozen_tokens: List[str] = [],
    ):
        (
            states_to_token_maps,
            empty_token_ids,
            fsm_finals,
        ) = _create_states_mapping_from_fsm(
            interegular_fsm, tokenizer, frozen_tokens=frozen_tokens
        )
        states_to_token_mask = {
            state: torch.tensor(list(next_tokens_to_end_states.keys()), device=device)
            for state, next_tokens_to_end_states in states_to_token_maps.items()
        }
        return cls(
            states_to_token_maps,
            empty_token_ids,
            fsm_finals,
            tokenizer.eos_token_id,
            states_to_token_mask,
        )

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the guide.

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """
        next_tokens_mask = self.states_to_token_mask.get(state)
        if next_tokens_mask is None:
            return Write(torch.tensor([self.eos_token_id]))

        return Generate(next_tokens_mask)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        We use the index to determine to which state the guide should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the guide.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the guide.

        """
        if token_id == self.eos_token_id or state not in self.states_to_token_maps:
            return -1

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            next_state = -1

        return next_state

    def is_final_state(self, state: int) -> bool:
        """Determine whether the current state of the guide is a final state."""
        return state in self.final_states

    def copy(self):
        return self


class RegexGuide(CoreRegexGuide):
    """
    Guide to generate text in the language of a regular expression.
    CoreRegexGuide with outlines cache
    """

    @classmethod
    def from_regex(  # type: ignore
        cls,
        regex_string: str,
        tokenizer,
        **kwargs,
    ):
        return super().from_regex(
            regex_string,
            tokenizer,
            _create_states_mapping=cached_create_states_mapping,
            **kwargs,
        )

    @classmethod
    def from_interegular_fsm(cls, fsm, tokenizer):
        pass


CFGState = collections.namedtuple("CFGState", ["parser_state", "prev_token"])


class CFGGuide(Guide):
    """Guide to generate text that is in the language of a context-free Lark grammar."""

    def __init__(self, cfg_string: str, tokenizer):
        """
        Construct the PartialLark parser and set the empty initial_state (PartialParserState)
        """
        warnings.warn(
            "Outlines' public *community-contributed* CFG structured generation is experimental. "
            "Please review https://dottxt-ai.github.io/outlines/latest/reference/generation/cfg#disclaimer"
        )

        self.cfg_string = cfg_string
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.parser = PartialLark(
            cfg_string,
            parser="lalr",
            import_paths=[grammars.GRAMMAR_PATH],
        )
        self.initial_state = CFGState(
            parser_state=self.parser.parse(""), prev_token=None
        )

    def get_next_instruction(self, state: CFGState) -> Instruction:
        """Return the next instruction for guided generation.

        Current lazy approach:
        - For each token in the vocabulary
          - create a copy of the parsers state
          - add the tokens to the parsers input text
          - if valid, add token to returned tokens

        Further refinements are necessary for performant text processing.

        Parameters
        ----------
        state
            The guides current PartialParserState, or None if complete

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """

        if state.parser_state is None:
            return Write(torch.tensor([self.eos_token_id]))

        valid_tokens = list(
            self.iter_valid_token_ids(state, self.tokenizer.vocabulary.values())
        )
        if len(valid_tokens) == 1:
            return Write(torch.tensor(valid_tokens))
        return Generate(torch.tensor(valid_tokens))

    def iter_valid_token_ids(
        self, state: CFGState, candidate_token_ids: list
    ) -> Generator[int, None, None]:
        """
        Iterate over the given token_ids and yield those that are valid for the current parser state.

        Parameters
        ----------
        parser_state
            The current state of the parser, or None if complete.
        token_ids
            The list of token ids to check for validity.

        Yields
        ------
        int
            Valid token ids.
        """
        if state.parser_state is None:
            yield self.eos_token_id
            return

        for token_id in candidate_token_ids:
            if token_id == self.eos_token_id:
                if self.can_terminate_state(state):
                    yield token_id
            else:
                try:
                    self._get_parser_state_token_applied(state, int(token_id))
                    yield token_id
                except (
                    ValueError,
                    EOFError,
                    UnexpectedToken,
                    UnexpectedCharacters,
                    DedentError,
                ):
                    pass

    def get_next_state(self, state: CFGState, token_id: int) -> CFGState:
        """
        Update the state of the guide.
        Decode the token_id, and calculate the new parser_state with the token applied.

        Parameters
        ----------
        state
            The guides current PartialParserState, or None if complete
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The guides new PartialParserState

        """
        if state.parser_state is None or token_id == self.eos_token_id:
            parser_state = None
        else:
            parser_state = self._get_parser_state_token_applied(state, int(token_id))
        return CFGState(parser_state=parser_state, prev_token=token_id)

    def _get_parser_state_token_applied(
        self, state: CFGState, token_id: int
    ) -> PartialParserState:
        """
        Don't mutate `parser_state`, copy to protect

        Get the token string
          - if first token in generation: tokenizer.decode (no leading whitespace)
          - else: normalized (with possibly leading whitespace)

        Don't allow empty ("") tokens, raise ValueError
        """
        parser_state = copy.copy(state.parser_state)  # prevent side effects

        # normalize
        if state.prev_token is None:
            new_token_str = self.tokenizer.decode([token_id])[0]
        else:
            prev_token_str = self.tokenizer.decode([[state.prev_token]])[0]
            combined_token_str = self.tokenizer.decode([[state.prev_token, token_id]])[
                0
            ]
            new_token_str = combined_token_str[len(prev_token_str) :]

        if new_token_str == "":
            raise ValueError("empty next token")

        # update parser with new token
        parser_state.lexer.state.text += new_token_str
        self.parser.parse_from_state(parser_state, is_end=False)

        return parser_state

    def is_final_state(self, state: CFGState) -> bool:
        # TODO: remove this method, use can_terminate_state and must_terminate_state
        # here and in RegexGuide per https://github.com/dottxt-ai/outlines/issues/885
        return self.can_terminate_state(state)

    def can_terminate_state(self, state: CFGState) -> bool:
        """Generation is allowed to terminate"""
        if state.parser_state is not None:
            try:
                copy.copy(state.parser_state).feed_eof()
            except UnexpectedToken:
                return False
        return True

    def must_terminate_state(self, state: CFGState) -> bool:
        """Generation must terminate, no legal continuations"""
        return state.parser_state is None or set(state.parser_state.accepts()).issubset(
            {"$END"}
        )

    def copy(self) -> "CFGGuide":
        """Create a copy of the Guide."""
        return CFGGuide(self.cfg_string, self.tokenizer)
