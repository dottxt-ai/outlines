import random

import pytest
import torch

from outlines.text.masks import create_char_set_mask, create_float_mask, create_int_mask


def test_int_mask():
    vocabulary = {"1": 0, "12": 1, "12a": 2, "a1": 3, "1.3": 4}

    mask = create_int_mask(vocabulary)
    assert torch.equal(mask, torch.tensor([True, True, False, False, False]))


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
        ".0": 9,
    }

    mask = create_float_mask(vocabulary)
    assert torch.equal(
        mask,
        torch.tensor([True, True, False, False, True, True, True, False, False, True]),
    )


def test_char_set_mask():
    vocabulary = {}
    with pytest.raises(ValueError, match="single characters"):
        create_char_set_mask(vocabulary, ["ab"])

    vocabulary = {"a": 0, "ab": 1, "abc": 2, "1": 3, "1_a": 4}
    mask = create_char_set_mask(vocabulary, ["a", "b", "1", "_"])
    assert torch.equal(mask, torch.tensor([True, True, False, True, True]))

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
    assert torch.equal(mask, torch.ones(12, dtype=torch.bool))

    mask = create_char_set_mask(vocabulary, ["a"])
    assert torch.equal(mask, torch.zeros(12, dtype=torch.bool))

    mask = create_char_set_mask(vocabulary, ["\n", "\r", "\t"])
    assert torch.equal(mask, torch.zeros(12, dtype=torch.bool))
