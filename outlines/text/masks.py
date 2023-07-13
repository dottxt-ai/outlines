import re
from typing import Dict, Iterable

import torch

__all__ = [
    "create_char_set_mask",
    "create_float_mask",
    "create_int_mask",
    "create_mask_from_regex",
]


def create_mask_from_regex(vocabulary: Dict[str, int], regex: str) -> torch.BoolTensor:
    """Create a token mask from a regex.

    Parameters
    ----------
    vocabulary
        A dictionary that contains a tokenizer's vocabulary as a map
        between tokens and their ids.
    regex
        The regex that tokens need to respect.

    """
    program = re.compile(regex)

    mask = torch.zeros(len(vocabulary), dtype=torch.bool)
    for token, token_id in vocabulary.items():
        if program.match(token) is not None:
            mask[token_id] = True

    return mask


def create_int_mask(vocabulary: Dict[str, int]) -> torch.BoolTensor:
    """Create a mask to generate signed integers."""
    mask = create_mask_from_regex(vocabulary, r"^[-+]?\d+$")

    return mask


def create_float_mask(vocabulary: Dict[str, int]) -> torch.BoolTensor:
    """Create a mask to generate signed floating point numbers."""
    mask = create_mask_from_regex(vocabulary, r"^[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)$")

    return mask


def create_char_set_mask(
    vocabulary: Dict[str, int], char_set: Iterable[str]
) -> torch.BoolTensor:
    """Create a mask to only generate characters in a given set.

    Parameters
    ----------
    vocabulary
        A dictionary that contains a tokenizer's vocabulary as a map
        between tokens and their ids.
    char_set
        An iterable that contains the valid single characters.

    """
    for char in char_set:
        if len(char) != 1:
            raise ValueError(
                "The `char_set` argument of `char_set_mask` can only contain single characters."
            )

    char_set = re.escape("".join(char_set))
    regex = "^[" + char_set + "]+$"
    mask = create_mask_from_regex(vocabulary, regex)
    return mask
