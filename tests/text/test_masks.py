import random

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from outlines.text.masks import create_char_set_mask, create_float_mask, create_int_mask


def test_int_mask():
    vocabulary = {"1": 0, "12": 1, "12a": 2, "a1": 3, "1.3": 4}

    mask = create_int_mask(vocabulary)
    assert_array_equal(mask, np.array([True, True, False, False, False]))


def test_float_mask():
    vocabulary = {
        "1": 0,
        "12": 1,
        "12a": 2,
        "a1": 3,
        "1.3": 4,
        "1.": 5,
        "0.": 6,
        "1.2.3": 7,
        ".": 8,
    }

    mask = create_float_mask(vocabulary)
    assert_array_equal(
        mask, np.array([True, True, False, False, True, True, True, False, True])
    )


def test_char_set_mask():
    vocabulary = {}
    with pytest.raises(ValueError, match="single characters"):
        create_char_set_mask(vocabulary, ["ab"])

    vocabulary = {"a": 0, "ab": 1, "abc": 2, "1": 3, "1_a": 4}
    mask = create_char_set_mask(vocabulary, ["a", "b", "1", "_"])
    assert_array_equal(mask, np.array([True, True, False, True, True]))

    vocabulary = {
        "\\": 0,
        "$": 1,
        ".": 2,
        "|": 3,
        "?": 4,
        "*": 5,
        "(": 6,
        ")": 7,
        "[": 8,
        "]": 9,
        "{": 10,
        "}": 11,
    }

    char_set = ["\\", "$", ".", "|", "?", "*", "(", ")", "[", "]", "{", "}"]
    random.shuffle(char_set)

    mask = create_char_set_mask(vocabulary, char_set)
    assert_array_equal(mask, np.ones(12, dtype=np.bool_))

    mask = create_char_set_mask(vocabulary, ["a"])
    assert_array_equal(mask, np.zeros(12, dtype=np.bool_))

    mask = create_char_set_mask(vocabulary, ["\n", "\r", "\t"])
    assert_array_equal(mask, np.zeros(12, dtype=np.bool_))
