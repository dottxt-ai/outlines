import pytest
import torch

import outlines
from outlines.generate import continuous
from outlines.generate.continuous import BatchMismatchError, SampleMismatchError
from outlines.samplers import MultinomialSampler

# model = outlines.models.transformers("Writer/palmyra-small")
model = outlines.models.transformers("openai-community/gpt2")


# Unit Generator

unit_sample_generator = outlines.generate.choice(
    model, ["Positive", "Negative"], sampler=MultinomialSampler(1)
)
unit_sample_generator = continuous(unit_sample_generator)

unit_prompt = "This is a prompt to test the slicing of `sequence_state`."

# Generator

num_samples = 3
mult_samples_generator = outlines.generate.choice(
    model, ["Positive", "Negative"], sampler=MultinomialSampler(num_samples)
)
mult_samples_generator = continuous(mult_samples_generator)

mult_prompts = [
    "[BATCH_1]This is a prompt to test the slicing of `sequence_state`.",
    "[BATCH_2]This is a prompt to test the slicing of `sequence_state`.",
    "[BATCH_3]This is a prompt to test the slicing of `sequence_state`.",
]


# [CASE_1]`batch size = 1`` & `num_samples = 1``


response_case_1 = unit_sample_generator(unit_prompt)


# Check that the key doesn't need to be a Tuple when `batch_size*num_samples=1`.
@pytest.mark.parametrize(
    "ids_size_key",
    [slice(0, 5, 1), slice(1, 5, 1), slice(2, 5, 1)],
)
def test_slice_ids_size_key(ids_size_key: slice):
    sliced_sequence_state = response_case_1[ids_size_key]
    assert list(sliced_sequence_state)[0] == unit_prompt[ids_size_key]


# Check cases where KV Cache is preserved with the right dimensions.
# REMINDER of the cases:
# **(1)** The `SequenceState` object has `batch_size == 1` and `num_samples == 1`.
# (2) Any `SequenceState` sliced in a way that `batch_size == 1` and `num_samples == 1`,
# however user must modify the generator to have `num_samples == 1`. It has to match the `SequenceState`
# or an exception `SampleMismatch` will be raised.
@pytest.mark.parametrize(
    "ids_size_key",
    [
        slice(0, 7, 1),
        slice(0, 16, 1),
        slice(0, 12, 1),
        slice(0, 8, 1),
    ],
)
def test_preserved_kv_cache_case_1(ids_size_key):
    should_preserve_kv_cache = response_case_1[ids_size_key]
    ids_stop = should_preserve_kv_cache.token_ids.shape[1] - 1
    for single_head_kv_cache_tuple in should_preserve_kv_cache.kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        assert (k_cache.shape[0], k_cache.shape[2]) == (
            1,
            ids_stop,
        )
        assert (v_cache.shape[0], v_cache.shape[2]) == (
            1,
            ids_stop,
        )


"""
# HOWEVER, there are some slices that don't allow to preserve the KV Cache even under (1) and (2).
# Check [NOTE] [SPECIAL CASE] sections for `token_level_slice_from_string_level_slice` utility in ``generate/continuous.py`.
# If the API encounter such slices, a warning that the KV Cache is not saved will be raised.
@pytest.mark.parametrize(
    "ids_size_key",
    [
        slice(0, 1, 1),
        slice(0, 2, 1),
        slice(0, 4, 1),
    ],
)
def test_warning_when_conditions_checked_but_impossible_slices_case_1(ids_size_key):
    with pytest.warns(UserWarning):
        response_case_1[ids_size_key]
"""

# `batch size = 1` || `num_samples = 1`

# [CASE_2] `batch size = 1`
responses_case_2 = mult_samples_generator(unit_prompt)


# Check that the key doesn't need to be a Tuple of length 3.
@pytest.mark.parametrize(
    "samples_key, ids_size_key",
    [(1, slice(0, 5, 1)), (0, slice(1, 5, 1)), (2, slice(2, 5, 1))],
)
def test_slice_ids_size_key_samples_key(samples_key: int, ids_size_key: slice):
    sliced_sequence_state = responses_case_2[samples_key, ids_size_key]
    assert list(sliced_sequence_state)[0] == unit_prompt[ids_size_key]


@pytest.mark.parametrize(
    "samples_key, ids_size_key",
    [
        (2, slice(0, 8, 1)),
        (2, slice(0, 16, 1)),
        (1, slice(0, 12, 1)),
    ],
)
def test_preserved_kv_cache_special_case_2(samples_key, ids_size_key):
    should_preserve_kv_cache = responses_case_2[samples_key, ids_size_key]
    ids_stop = should_preserve_kv_cache.token_ids.shape[1] - 1
    for single_head_kv_cache_tuple in should_preserve_kv_cache.kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        assert (k_cache.shape[0], k_cache.shape[2]) == (
            1,
            ids_stop,
        )
        assert (v_cache.shape[0], v_cache.shape[2]) == (
            1,
            ids_stop,
        )


"""
@pytest.mark.parametrize(
    "samples_key, ids_size_key",
    [
        (1, slice(0, 4, 1)),
        (0, slice(0, 3, 1)),
    ],
)
def test_warning_when_conditions_checked_but_impossible_slices_case_2(
    samples_key, ids_size_key
):
    with pytest.warns(UserWarning):
        responses_case_2[samples_key, ids_size_key]
"""

# [CASE 3] `num_samples = 1`
responses_case_3 = unit_sample_generator(mult_prompts)


# Check that the key doesn't need to be a Tuple of length 3.
@pytest.mark.parametrize(
    "batch_key, ids_size_key",
    [(1, slice(0, 5, 1)), (0, slice(1, 5, 1)), (2, slice(2, 5, 1))],
)
def test_slice_ids_size_key_batch_key(batch_key: int, ids_size_key: slice):
    sliced_sequence_state = responses_case_3[batch_key, ids_size_key]
    assert list(sliced_sequence_state)[0] == mult_prompts[batch_key][ids_size_key]


@pytest.mark.parametrize(
    "batch_key, ids_size_key",
    [
        (1, slice(0, 4, 1)),
        (0, slice(0, 2, 1)),
        (2, slice(0, 8, 1)),
        (2, slice(0, 16, 1)),
    ],
)
def test_preserved_kv_cache_special_case_3(batch_key: int, ids_size_key: slice):
    should_preserve_kv_cache = responses_case_3[batch_key, ids_size_key]
    ids_stop = should_preserve_kv_cache.token_ids.shape[1] - 1
    for single_head_kv_cache_tuple in should_preserve_kv_cache.kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        assert (k_cache.shape[0], k_cache.shape[2]) == (
            1,
            ids_stop,
        )
        assert (v_cache.shape[0], v_cache.shape[2]) == (
            1,
            ids_stop,
        )


"""
@pytest.mark.parametrize(
    "batch_key, ids_size_key",
    [
        (0, slice(0, 1, 1)),
        (1, slice(0, 12, 1)),
    ],
)
def test_warning_when_conditions_checked_but_impossible_slices_case_3(
    batch_key: int, ids_size_key: slice
):
    with pytest.warns(UserWarning):
        responses_case_3[batch_key, ids_size_key]
"""

# `batch size = 1` & `num_samples = 1`

responses = mult_samples_generator(mult_prompts)

# Check that the slicing is consistent with the prompt.

# Check[1]: batch_key: slice, samples_key: slice, id_key: slice


@pytest.mark.parametrize(
    "batch_key, samples_key, ids_size_key",
    [
        (slice(0, 2, 1), slice(2, 3, 1), slice(2, 4, 1)),
        (slice(2, 3, 1), slice(0, 2, 1), slice(2, 8, 1)),
        (slice(0, 3, 1), slice(0, 3, 1), slice(4, 8, 1)),
    ],
)
def test_slice_batch_key_slice_samples_key_slice_ids_size_key(
    batch_key: slice, samples_key: slice, ids_size_key: slice
):
    sliced_sequence_state = iter(list(responses[batch_key, samples_key, ids_size_key]))
    for batch_idx in range(batch_key.start, batch_key.stop):
        for sample_idx in range(samples_key.start, samples_key.stop):
            prompt_idx = (num_samples * batch_idx + sample_idx) // num_samples
            assert next(sliced_sequence_state) == mult_prompts[prompt_idx][ids_size_key]


# Check[2]: batch_key: int, samples_key: slice, id_key: slice
@pytest.mark.parametrize(
    "batch_key, samples_key, ids_size_key",
    [
        (0, slice(2, 3, 1), slice(2, 4, 1)),
        (1, slice(0, 2, 1), slice(2, 8, 1)),
        (2, slice(0, 3, 1), slice(4, 8, 1)),
    ],
)
def test_int_batch_key_slice_samples_key_slice_ids_size_key(
    batch_key: int, samples_key: slice, ids_size_key: slice
):
    sliced_sequence_state = iter(list(responses[batch_key, samples_key, ids_size_key]))
    for _ in range(samples_key.start, samples_key.stop):
        assert next(sliced_sequence_state) == mult_prompts[batch_key][ids_size_key]


# Check[3]: batch_key: slice, samples_key: int, id_key: slice
@pytest.mark.parametrize(
    "batch_key, samples_key, ids_size_key",
    [
        (slice(0, 2, 1), 1, slice(2, 4, 1)),
        (slice(2, 3, 1), 2, slice(2, 8, 1)),
        (slice(0, 3, 1), 0, slice(4, 8, 1)),
    ],
)
def test_slice_batch_key_int_samples_key_slice_ids_size_key(
    batch_key: slice, samples_key: int, ids_size_key: slice
):
    sliced_sequence_state = iter(list(responses[batch_key, samples_key, ids_size_key]))
    for batch_idx in range(batch_key.start, batch_key.stop):
        assert next(sliced_sequence_state) == mult_prompts[batch_idx][ids_size_key]


# Check[4]: batch_key: int, samples_key: int, id_key: slice
@pytest.mark.parametrize(
    "batch_key, samples_key, ids_size_key",
    [
        (0, 1, slice(2, 4, 1)),
        (1, 2, slice(2, 8, 1)),
        (2, 0, slice(4, 8, 1)),
    ],
)
def test_int_batch_key_int_samples_key_slice_ids_size_key(
    batch_key: int, samples_key: slice, ids_size_key: slice
):
    sliced_sequence_state = responses[batch_key, samples_key, ids_size_key]
    assert list(sliced_sequence_state)[0] == mult_prompts[batch_key][ids_size_key]


# Check cases where KV Cache is preserved with the right dimensions.
# REMINDER of the cases:
# (1) The `SequenceState` object has `batch_size == 1` and `num_samples == 1`.
# **(2)** Any `SequenceState` sliced in a way that `batch_size == 1` and `num_samples == 1`,
# however user must modify the generator to have `num_samples == 1`. It has to match the `SequenceState`
# or an exception `SampleMismatch` will be raised.
@pytest.mark.parametrize(
    "batch_key, samples_key, ids_size_key",
    [
        (0, 1, slice(0, 4, 1)),
        (1, 2, slice(0, 8, 1)),
        (0, 2, slice(0, 16, 1)),
        (0, 0, slice(0, 4, 1)),
        (1, 1, slice(0, 8, 1)),
    ],
)
def test_preserved_kv_cache(batch_key, samples_key, ids_size_key):
    should_preserve_kv_cache = responses[batch_key, samples_key, ids_size_key]
    ids_stop = should_preserve_kv_cache.token_ids.shape[1] - 1
    for single_head_kv_cache_tuple in should_preserve_kv_cache.kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        assert (k_cache.shape[0], k_cache.shape[2]) == (
            1,
            ids_stop,
        )
        assert (v_cache.shape[0], v_cache.shape[2]) == (
            1,
            ids_stop,
        )


"""
# HOWEVER, there are some slices that don't allow to preserve the KV Cache even under (1) and (2).
# Check [NOTE] [SPECIAL CASE] sections for `token_level_slice_from_string_level_slice` utility in ``generate/continuous.py`.
# If the API encounter such slices, a warning that the KV Cache is not saved will be raised.


@pytest.mark.parametrize(
    "batch_key, samples_key, ids_size_key",
    [
        (0, 2, slice(0, 1, 1)),
        (2, 2, slice(0, 12, 1)),
    ],
)
def test_warning_when_conditions_checked_but_impossible_slices(
    batch_key, samples_key, ids_size_key
):
    with pytest.warns(UserWarning):
        responses[batch_key, samples_key, ids_size_key]
"""


# Check cases where KV Cache should be reinitialized.
@pytest.mark.parametrize(
    "batch_len, samples_len, ids_stop",
    [
        (slice(None, None, None), slice(None, None, None), slice(0, 4, 1)),
        (slice(None, None, None), slice(None, None, None), slice(0, 8, 1)),
        (slice(2, 3, 1), slice(0, 2, 1), slice(0, 12, 1)),
        (slice(0, 3, 1), slice(0, 3, 1), slice(0, 16, 1)),
        (0, slice(2, 3, 1), slice(0, 4, 1)),
        (slice(0, 2, 1), 1, slice(0, 8, 1)),
    ],
)
def test_not_preserved_kv_cache(batch_len, samples_len, ids_stop):
    should_not_preserve_kv_cache = responses[batch_len, samples_len, ids_stop]
    assert should_not_preserve_kv_cache.kv_cache is None


# Check that the weights accumulate (same SequenceGenerator) with nested runs.
def test_weight_accumulation_same_sequence_generator():
    assert torch.all(mult_samples_generator(responses).weights < responses.weights)


# Check that the weights accumulate with type of generation (from `choice` to `text`).
def test_weight_accumulation_different_type_generation():
    generator = outlines.generate.text(model, sampler=MultinomialSampler(num_samples))
    generator = continuous(generator)
    assert torch.all(generator(responses, max_tokens=10).weights < responses.weights)


# Check that weights of two added sequences is summed.
def test_weight_accumulation_added_sequence_generators():
    other_responses = mult_samples_generator(responses)
    assert torch.all(
        (responses + other_responses).weights
        == (responses.weights + other_responses.weights)
    )


# Check that weights should be reinitialized when adding a string.
def test_weight_accumulation_added_string():
    assert torch.all((response_case_1 + "a test string.").weights == 0)
    assert torch.all(
        (
            responses
            + ["a test string." for _ in range(len(mult_prompts) * num_samples)]
        ).weights
        == 0
    )


# Check Exception if num_samples changes.
@pytest.mark.parametrize(
    "num_samples",
    [1, 5, 8],
)
def test_exception_if_num_samples_changes(num_samples: int):
    with pytest.raises(SampleMismatchError) as exc_info:
        generator = outlines.generate.choice(
            model, ["Positive", "Negative"], sampler=MultinomialSampler(num_samples)
        )
        generator = continuous(generator)
        generator(responses)
    assert (
        str(exc_info.value)
        == f"Continuous generation can't proceed, Generator has a `num_samples == {num_samples}` and SequenceState has a `num_samples == {responses.weights.shape[1]}`. \
            A new generator with `num_samples == {responses.weights.shape[1]}` should be utilized to proceed."
    )


# Check Exception if added Sequence and str have different `batch_size`.
def test_exception_added_sequence_state_str_batch_mismatch():
    with pytest.raises(BatchMismatchError) as exc_info:
        responses + "This is a test prompt."
    assert (
        str(exc_info.value)
        == "A sequence and a string were added and their batch sizes were different."
    )


# Check Exception if added Sequence and Sequence have different `batch_size`.
def test_exception_added_sequence_state_sequence_state_batch_mismatch():
    with pytest.raises(BatchMismatchError) as exc_info:
        responses + responses[:1, :1, :]
    assert (
        str(exc_info.value)
        == "Sequences were added and their batch sizes were different."
    )
