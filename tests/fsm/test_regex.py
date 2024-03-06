import interegular
import numba
import pytest
from transformers import AutoTokenizer

from outlines.fsm.regex import (
    _walk_fsm,
    create_fsm_index_end_to_end,
    create_fsm_index_tokenizer,
    fsm_union,
    get_sub_fsms_from_seq,
    make_deterministic_fsm,
    walk_fsm,
)
from outlines.models.transformers import TransformerTokenizer


def walk_fsm_numba(
    fsm,
    input_string: str,
    start_state: int,
    full_match: bool = True,
):
    return _walk_fsm(
        fsm.fsm_info.transitions,
        fsm.fsm_info.alphabet_symbol_mapping,
        fsm.fsm_info.alphabet_anything_value,
        fsm.fsm_info.initial,
        fsm.fsm_info.finals,
        input_string,
        start_state,
        full_match=full_match,
    )


@pytest.mark.parametrize(
    "function",
    [
        walk_fsm,
        walk_fsm_numba,
    ],
)
def test_walk_fsm(function):
    regex_pattern = interegular.parse_pattern("0|[1-9][2-9]*")
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    res = tuple(function(regex_fsm, "0", regex_fsm.initial, full_match=True))
    assert res == (1,)

    res = tuple(function(regex_fsm, "00", regex_fsm.initial, full_match=False))
    assert res == (1,)

    res = tuple(function(regex_fsm, "!", regex_fsm.initial, full_match=True))
    assert res == tuple()

    res = tuple(function(regex_fsm, "00", regex_fsm.initial, full_match=True))
    assert res == tuple()

    # This should fail, because state `1` reads nothing
    res = tuple(function(regex_fsm, "0", 1, full_match=True))
    assert res == tuple()

    regex_pattern = interegular.parse_pattern("0|[1-9][2-9]+")
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    res = tuple(function(regex_fsm, "1", regex_fsm.initial, full_match=True))
    assert res == tuple()

    res = tuple(function(regex_fsm, "1", regex_fsm.initial, full_match=False))
    assert res == (2,)

    res = tuple(function(regex_fsm, "12", regex_fsm.initial, full_match=True))
    assert res == (2, 3)

    pattern = interegular.parse_pattern(r"(?:[^\W\d]\w*|[\t \x0c]+)")
    fsm, _ = make_deterministic_fsm(pattern.to_fsm().reduce())

    res = tuple(function(fsm, "x ", fsm.initial, full_match=False))
    assert res == (2,)

    start_state = list(fsm.finals)[0]
    res = tuple(function(fsm, "!", start_state, full_match=False))
    assert res == tuple()


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

    state_seq = walk_fsm(fsm, "def", fsm.initial)
    state_seq.insert(0, fsm.fsm_info.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, False, True), (2, True, True)]

    # Make sure the old-to-new state map is correct
    def_state_seq = walk_fsm(def_fsm, "def", fsm.initial)
    def_state_seq.insert(0, fsm.fsm_info.initial)

    def_old_to_new_states = fsms_to_trans_finals[0][2]
    assert all(
        new_state in def_old_to_new_states[old_state]
        for old_state, new_state in zip(def_state_seq, state_seq)
    )

    state_seq = walk_fsm(fsm, "ef", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(2, True, True)]

    name_state_seq = walk_fsm(name_fsm, "ef", fsm.initial)
    name_state_seq.insert(0, fsm.initial)

    name_old_to_new_states = fsms_to_trans_finals[2][2]
    assert all(
        new_state in name_old_to_new_states[old_state]
        for old_state, new_state in zip(name_state_seq, state_seq)
    )

    state_seq = walk_fsm(fsm, "match", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(1, False, True), (2, True, True)]

    match_state_seq = walk_fsm(match_fsm, "match", fsm.initial)
    match_state_seq.insert(0, fsm.initial)

    match_old_to_new_states = fsms_to_trans_finals[1][2]
    assert all(
        new_state in match_old_to_new_states[old_state]
        for old_state, new_state in zip(match_state_seq, state_seq)
    )

    state_seq = walk_fsm(fsm, "defa", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(2, True, True)]

    state_seq = walk_fsm(fsm, "de", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, True, False), (2, True, True)]

    state_seq = walk_fsm(fsm, "+", fsm.initial, False)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(3, True, False), (4, False, True)]

    state_seq = walk_fsm(fsm, "+=", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(3, False, True)]

    # Test some overlapping patterns
    join_fsms = [
        interegular.parse_pattern(r"JOIN").to_fsm().reduce(),
        interegular.parse_pattern(r"JOIN LEFT").to_fsm().reduce(),
    ]
    fsm, fsms_to_trans_finals = fsm_union(join_fsms)

    # Matching "OI"
    state_seq = [1, 2, 3]
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, True, False), (1, True, False)]

    # Matching "N"
    state_seq = [3, 4]
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, False, True), (1, True, False)]

    # Matching " "
    state_seq = [4, 5]
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(1, True, False)]


def test_create_fsm_index_end_to_end():
    regex_str = "0|[1-9][0-9]*"

    regex_pattern = interegular.parse_pattern(regex_str)
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    vocabulary = {
        "blah": numba.typed.List([0]),
        "1a": numba.typed.List([1]),
        "2": numba.typed.List([2]),
        "0": numba.typed.List([3]),
        "<EOS>": numba.typed.List([4]),
    }

    vocabulary_nb = numba.typed.Dict.empty(
        numba.types.string, numba.types.ListType(numba.int64)
    )
    vocabulary_nb.update(vocabulary)

    res = create_fsm_index_end_to_end(regex_fsm.fsm_info, vocabulary_nb)

    assert res == {0: {(2, 2), (3, 1)}, 2: {(2, 2), (3, 2)}}


def test_create_fsm_index_tokenizer():
    # The combined regular expressions of a lexer state in a Python grammar
    regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

    regex_pattern = interegular.parse_pattern(regex_str)
    # Not reduced, so that there are many states
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm())

    num_fsm_states = len(regex_fsm.states)
    assert num_fsm_states == 220

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = TransformerTokenizer(tokenizer)

    states_to_token_subsets, empty_token_ids = create_fsm_index_tokenizer(
        regex_fsm, tokenizer
    )

    assert not empty_token_ids
    assert len(states_to_token_subsets) / num_fsm_states > 0.94


@pytest.mark.skip(reason="Only for local profiling")
def test_regex_index_performance():
    from line_profiler import LineProfiler  # type: ignore [import]

    regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

    regex_pattern = interegular.parse_pattern(regex_str)
    # Not reduced, so that there are many states
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm())

    num_fsm_states = len(regex_fsm.states)
    assert num_fsm_states == 220

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = TransformerTokenizer(tokenizer)

    # Pre-compile Numba functions
    res, _ = create_fsm_index_tokenizer(regex_fsm, tokenizer)
    assert len(res) > 1

    profiler = LineProfiler(create_fsm_index_end_to_end)

    profiler.runctx(
        "create_fsm_index_tokenizer(regex_fsm, tokenizer)",
        globals(),
        locals(),
    )
    profiler.dump_stats("line-profiler-create_fsm_index.pkl")
    profiler.print_stats(output_unit=1e-3, summarize=True, stripzeros=True)


@pytest.mark.skip(reason="Only for local profiling")
def test_json_index_performance():
    import json
    from enum import Enum

    from line_profiler import LineProfiler  # type: ignore [import]
    from pydantic import BaseModel, constr

    import outlines

    class Weapon(str, Enum):
        sword = "sword"
        axe = "axe"
        mace = "mace"
        spear = "spear"
        bow = "bow"
        crossbow = "crossbow"

    class Armor(str, Enum):
        leather = "leather"
        chainmail = "chainmail"
        plate = "plate"

    class Character(BaseModel):
        name: constr(max_length=10)
        # TODO: Add support for conint
        age: int  # conint(int, ge=18, le=100)
        armor: Armor
        weapon: Weapon
        # TODO: Add support for conint
        strength: int  # conint(int, ge=0, le=100)

    model = outlines.models.transformers("gpt2", device="cuda")
    json_schema = json.dumps(Character.model_json_schema())

    def build_regex():
        regex_str = outlines.index.json_schema.build_regex_from_object(json_schema)
        outlines.generate.regex(model, regex_str)

    profiler = LineProfiler(create_fsm_index_end_to_end)
    profiler.add_function(create_fsm_index_tokenizer)
    profiler.add_function(outlines.index.index.RegexFSM.__init__)

    profiler.runctx(
        "build_regex()",
        globals(),
        locals(),
    )
    profiler.dump_stats("line-profiler-build-json-regex.pkl")
    profiler.print_stats(output_unit=1e-3, summarize=True, stripzeros=True)
