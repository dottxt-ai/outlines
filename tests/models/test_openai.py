import pytest

from outlines.models.openai import (
    build_optimistic_mask,
    find_longest_intersection,
    find_response_choices_intersection,
)


@pytest.mark.parametrize(
    "response,choice,expected_intersection,expected_choices_left",
    (
        ([1, 2, 3, 4], [[5, 6]], [], [[5, 6]]),
        ([1, 2, 3, 4], [[5, 6], [7, 8]], [], [[5, 6], [7, 8]]),
        ([1, 2, 3, 4], [[1, 2], [7, 8]], [1, 2], [[]]),
        ([1, 2], [[1, 2, 3, 4], [1, 2]], [1, 2], [[3, 4], []]),
        ([1, 2, 3], [[1, 2, 3, 4], [1, 2]], [1, 2, 3], [[4]]),
    ),
)
def test_find_response_choices_intersection(
    response, choice, expected_intersection, expected_choices_left
):
    intersection, choices_left = find_response_choices_intersection(response, choice)
    assert intersection == expected_intersection
    assert choices_left == expected_choices_left


@pytest.mark.parametrize(
    "response,choice,expected_prefix",
    (
        ([1, 2, 3], [1, 2, 3, 4], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3], [1, 2, 3]),
        ([4, 5], [1, 2, 3], []),
    ),
)
def test_find_longest_common_prefix(response, choice, expected_prefix):
    prefix = find_longest_intersection(response, choice)
    assert prefix == expected_prefix


@pytest.mark.parametrize(
    "transposed,mask_size,expected_mask",
    (
        ([{1, 2}, {3, 4}], 3, {1: 100, 2: 100, 3: 100}),
        ([{1, 2}, {3, 4}], 4, {1: 100, 2: 100, 3: 100, 4: 100}),
    ),
)
def test_build_optimistic_mask(transposed, mask_size, expected_mask):
    mask = build_optimistic_mask(transposed, mask_size)
    assert mask == expected_mask
