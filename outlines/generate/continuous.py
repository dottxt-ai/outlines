import copy
import dataclasses
import warnings
from typing import List, Optional, Tuple, Union

import torch

from outlines.generate.api import SequenceGenerator
from outlines.generate.generator import sequence_generator
from outlines.models.tokenizer import Tokenizer

warnings.simplefilter("always", UserWarning)


# Thrown when two added sequences have different batch size.
class BatchMismatchError(Exception):
    pass


# Thrown when (1) user slices the samples,
# then uses the generator (set to the original number of samples) to continue generation.
# (2) Two added sequences have different number of samples.
class SampleMismatchError(Exception):
    pass


# Handeled to reset the KV Cache, thrown after reaching some slices, under which,
# it's not possible to save the KV Cache.
class SlicingError(Exception):
    pass


# [NOTE] Python doesn't allow using `Union[int, List[int]]` with `isinstance``,
# nor `(isinstance(prompt_or_sequence_state, int) or isinstance(prompt_or_sequence_state, List[int])`.
def islist(
    variable,
    type,
) -> bool:
    return isinstance(variable, list) and all(
        isinstance(item, type) for item in variable
    )


# Decouples the batch and the samples using sublists.
def group_samples_into_sublists(
    batch_and_samples: List[str], num_samples: int
) -> List[List[str]]:
    num_sublists = len(batch_and_samples) // num_samples
    return [
        batch_and_samples[i * num_samples : (i + 1) * num_samples]
        for i in range(num_sublists)
    ]


# Make sure that `.start` and `.stop` are defined in a slice when
# `N:`, `:M` or `:` is used.
def make_sure_slice_start_stop_defined(key: slice, start: int, stop: int):
    if key.start is None:
        key = slice(start, key.stop, None)
    if key.stop is None:
        key = slice(
            key.start,
            stop,
            None,
        )
    return key


# ------------------------------- Utilities for KV Cache -------------------------------
# HuggingFace using `Tuple[Tuple[torch.Tensor]]` instead of a plain `torch.Tensor`, will force us
# to each time iterate through the tuples to do any torch operation. We also need to create
# a new `Tuple[Tuple[torch.Tensor]]` since it's immutable.
# Unfortunately, this will result in boilerplate code.

# This is how KV Cache is structured from HuggingFace documentation:
# Type: tuple(tuple(torch.FloatTensor))
# [NOTE] The first Tuple contains the attention heads, the second one contains the keys and values.
# Structure: (batch_size, num_heads, sequence_length, embed_size_per_head)

# IMPORTANT: `logits, kv_cache = model(token_ids, attention_masks, kv_cache)` imposes that `kv_cache`
# is either None or a **tensor (in each attention head, each key and value) with a sequence length less than one than the one of the token_ids**.


# Decouples the first `batch_key*samples_key` dimension into `batch_key` and `samples_key` dimensions.
# [NOTE] Makes slicing of `kv_cache` simpler when using both `batch_key` and `sample_key`.
def rearrange_kv_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    batch_size: int,
    num_samples: int,
):
    if kv_cache is None:
        return None

    rearranged_kv_cache = []
    for single_head_kv_cache_tuple in kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        sliced_k_cache = k_cache.view(
            batch_size,
            num_samples,
            *k_cache.shape[1:],
        )
        sliced_v_cache = v_cache.view(
            batch_size,
            num_samples,
            *v_cache.shape[1:],
        )
        rearranged_kv_cache.append((sliced_k_cache, sliced_v_cache))
    return tuple(rearranged_kv_cache)


# Rearranges the first dimension back to its expanded version `batch_key*sample_key`.
def expand_kv_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
):
    if kv_cache is None:
        return None

    expanded_kv_cache = []
    for single_head_kv_cache_tuple in kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        sliced_k_cache = k_cache.view(
            -1,
            *k_cache.shape[2:],
        )
        sliced_v_cache = v_cache.view(
            -1,
            *v_cache.shape[2:],
        )
        expanded_kv_cache.append((sliced_k_cache, sliced_v_cache))
    return tuple(expanded_kv_cache)


def slice_kv_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ids_size_key_stop: int,
    batch_key_mult_samples_key: Union[int, slice],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if kv_cache is None:
        return None

    ids_size_key_stop = ids_size_key_stop if ids_size_key_stop != 0 else 1

    sliced_kv_cache = []
    for single_head_kv_cache_tuple in kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        sliced_k_cache = k_cache[batch_key_mult_samples_key, :, :ids_size_key_stop, :]
        sliced_v_cache = v_cache[batch_key_mult_samples_key, :, :ids_size_key_stop, :]
        # Dealing with PyTorch's (int) indexing which removes a dimension.
        if isinstance(batch_key_mult_samples_key, int):
            sliced_k_cache, sliced_v_cache = sliced_k_cache.unsqueeze(
                0
            ), sliced_v_cache.unsqueeze(0)
        sliced_kv_cache.append((sliced_k_cache, sliced_v_cache))
    return tuple(sliced_kv_cache)


# [NOTE] Since `sequence_generator` deals with flattened `num_samples*batch_size` dimension,
# rearranging will make slicing such as [batch_key: slice, samples_key: slice, ...] possible.
def slice_while_rearranged_kv_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ids_size_key_stop: int,
    batch_key: Union[int, slice],
    samples_key: Union[int, slice],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if kv_cache is None:
        return None

    ids_size_key_stop = ids_size_key_stop if ids_size_key_stop != 0 else 1

    sliced_kv_cache = []
    for single_head_kv_cache_tuple in kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        sliced_k_cache = k_cache[batch_key, samples_key, :, :ids_size_key_stop, :]
        sliced_v_cache = v_cache[batch_key, samples_key, :, :ids_size_key_stop, :]
        # Dealing with PyTorch's (int) indexing which removes a dimension.
        if isinstance(batch_key, int):
            sliced_k_cache, sliced_v_cache = sliced_k_cache.unsqueeze(
                0
            ), sliced_v_cache.unsqueeze(0)
        if isinstance(samples_key, int):
            sliced_k_cache, sliced_v_cache = sliced_k_cache.unsqueeze(
                0
            ), sliced_v_cache.unsqueeze(0)
        sliced_kv_cache.append((sliced_k_cache, sliced_v_cache))
    return tuple(sliced_kv_cache)


# Just applies `torch.roll` to `kv_cache` (shifts element in some dimension),
# see https://pytorch.org/docs/stable/generated/torch.roll.html.
# Shifts the last element in the `sequence_length` of `kv_cache` to the beginning.
def roll_kv_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    batch_key_mult_num_samples_key: int,
    shift: int = 1,
    dim: int = 1,
):
    if kv_cache is None:
        return None

    for single_head_kv_cache_tuple in kv_cache:
        k_cache, v_cache = single_head_kv_cache_tuple
        k_cache[batch_key_mult_num_samples_key, :, :, :] = torch.roll(
            k_cache[batch_key_mult_num_samples_key, :, :, :], shifts=shift, dims=dim
        )
        v_cache[batch_key_mult_num_samples_key, :, :, :] = torch.roll(
            v_cache[batch_key_mult_num_samples_key, :, :, :], shifts=shift, dims=dim
        )
    return kv_cache


# ------------------------------- Utilities for KV Cache -------------------------------


# Finds the intersection between two strings, example:
# String 1: Guided Generation is cool.
# String 2: Guided Generation is cool. Try it!
# Intersection: Guided Generation is cool.
def find_word_intersection(str1: str, str2: str):
    return str1[: str1.find(str2) + len(str2)]


# Order matters here, it's str2 + rest = str1.
def find_rest_from_str(str1: str, str2: str):
    return str1.replace(str2, "", 1)


# Why?
# The KV Cache is arranged on TOKEN-LEVEL indices, users slice the prompt and not the `token_ids`.
# When saving the KV Cache during a slice, specifically the case where `token_ids_key.start = 0`.
# We need to convert the STR-LEVEL slice (from users) to a TOKEN-LEVEL slice, then use it to
# slice the KV Cache.

# How?
# We incrementally reduce token by token from token_ids, we then check if the
# the reduced `one_token_less_token_ids` didn't go over the slicing limit.
# If, in the process, it doesn't and at the same time, it is equal to the `sliced_prompt`,
# we recover the TOKEN-LEVEL slice.
# If it doesn't during the loop, we'll keep going until we go over the limit and then get the rest between
# `sliced_prompt` and `one_token_less_prompt`, tokenize it, and finally concatenate it
# in the TOKEN-LEVEL with `one_token_less_token_ids`.

# [NOTE] One issue that could happen is when the rest is more than one token.
# If not sure why this is an issue, check the first paragraph under 'Utilities for KV Cache' section above.
# In this case, we'll recompute the KV Cache.

# [NOTE] Slicing through a whole batch introduces even more complexity, because each element
# in the batch will be sliced at a different index on the TOKEN-LEVEL. KV Cache will not be saved in this case.
# To implement it, one must understand how KV Cache deals with the padding.


def token_level_slice_from_string_level_slice(
    sliced_prompt: str, token_ids: torch.tensor, tokenizer: "Tokenizer"
):
    one_token_less_token_ids = token_ids[0, :-1]
    one_token_less_prompt = "".join(tokenizer.decode(one_token_less_token_ids))

    intersection = find_word_intersection(sliced_prompt, one_token_less_prompt)

    # [NOTE] If the condition `sliced_prompt == intersection` is False, we've gone over the slicing limit.
    # Example:
    # Full prompt: "Guided generation is awesome!", Full tokens: [45, 65, 192, 345, 125]
    # User's slice: "Guided generation i"
    # Reached slice: "Guided genera", tokens: [45, 65, 192]
    # We compute the rest, here is 'tion i', tokenize it and concatenate to [45, 65, 192].
    while sliced_prompt == intersection:
        # Deals with the special case where the slice on the str-level stops at a token-level index and not inside.
        # Example:
        # Full prompt: "Guided generation is awesome!", Full tokens: [45, 65, 192, 345, 125]
        # User's slice: "Guided generation is"
        # Reached slice: "Guided generation is", tokens: [45, 65, 192, 345]
        if sliced_prompt == one_token_less_prompt:
            # The slice doesn't stop inside a token, returns where it should be sliced on token-level.
            # Returns `ids_size_key.stop` on token-level.
            # KV Cache is one token less than token_ids size.
            kv_cache_stop_index = one_token_less_token_ids.shape[0] - 1
            # [NOTE] [SPECIAL CASE] If the slice keeps one single token from `token_ids`, then the KV Cache
            # should be reinitialized.
            if kv_cache_stop_index == 0:
                raise SlicingError(
                    "It is not possible save the KV Cache with this slicing."
                )
            else:
                return kv_cache_stop_index, one_token_less_token_ids.unsqueeze(0)
        one_token_less_token_ids = one_token_less_token_ids[:-1]
        one_token_less_prompt = "".join(tokenizer.decode(one_token_less_token_ids))
        intersection = find_word_intersection(sliced_prompt, one_token_less_prompt)

    # Turn the rest into tokens and then concatenate with `one_token_less_token_ids`.
    prompt_rest = find_rest_from_str(sliced_prompt, one_token_less_prompt)
    token_ids_rest = tokenizer.encode(prompt_rest)[0]

    # [NOTE] [SPECIAL CASE] If the rest is more than one token, then KV Cache should be recomputed.
    # HuggingFace's implementation involves a shift of 1 token in the context window between the logits and the KV Cache.
    if token_ids_rest.shape[1] > 1:
        # print(f"Slice has a rest of {token_ids_rest.shape[1]} tokens")
        raise SlicingError("It is not possible to save the KV Cache with this slicing.")

    final_token = torch.cat(
        (one_token_less_token_ids.unsqueeze(0), token_ids_rest), dim=1
    )

    # [NOTE] `kv_cache` SHOULD be one token less than `token_ids` size.
    kv_cache_stop_index = final_token.shape[1] - 1

    # [NOTE] [SPECIAL CASE] If the slice keeps one single token from `token_ids`, then the KV Cache
    # should be reinitialized.
    if kv_cache_stop_index == 0:
        raise SlicingError("It is not possible to save the KV Cache with this slicing.")
    else:
        return kv_cache_stop_index, final_token


# ------------------------- NOT IMPLEMENTED: SLOW -------------------------
# Why?
# The reason comes from the way HuggingFace's model deals with the KV Cache.
# `logits, kv_cache = model(token_ids, attention_masks, kv_cache)` imposes that `kv_cache`
# is either (1) None or (2) a **tensor (in each attention head, each key and value)
# with a sequence length less than one than the one of the token_ids**.

# REMINDER:
# This is how KV Cache is structured from HuggingFace documentation:
# Type: tuple(tuple(torch.FloatTensor))
# [NOTE] The first Tuple contains the attention heads, the second one contains the keys and values.
# Structure: (batch_size, num_heads, sequence_length, embed_size_per_head)

# This deals with the cases (1) adding two `SequenceState`
# and (2) a slice that results with a rest (see `token_level_slice_from_string_level_slice`) of more than one token.
# This enables to:
# 1) Keep the KV Cache of the first sequence and **then completes the part of the other sequence**
# when adding two sequences.
# 2) Keep some part of the KV Cache when the conditions (1) and (2) are met but the slice doesn't,
# see `token_level_slice_from_string_level_slice`.

# [NOTE] From tests I've conducted, this is slower than recomputing it.


def complete_kv_cache_from_token_ids(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    token_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    model,
):
    if kv_cache is None:
        return token_ids, None

    _, token_ids_stop = token_ids.shape

    init_kv_cache_stop = kv_cache[0][0].shape[2]

    if token_ids_stop != init_kv_cache_stop:
        kv_cache_stop = init_kv_cache_stop
        while token_ids_stop != kv_cache_stop + 1:
            # [QUESTION] Not an optimal way to compute the KV Cache, can we do this without computing the logits?
            _, kv_cache = model(
                token_ids[:, : kv_cache_stop + 1],
                attention_masks[:, : kv_cache_stop + 1],
                kv_cache,
            )
            kv_cache_stop = kv_cache[0][0].shape[2]

        return token_ids, kv_cache
    else:
        return token_ids, kv_cache


# ------------------------- NOT IMPLEMENTED: SLOW -------------------------


@dataclasses.dataclass
class SequenceState:
    token_ids: torch.Tensor
    weights: torch.Tensor
    attention_masks: torch.Tensor
    kv_cache: torch.Tensor
    tokenizer: "Tokenizer"

    def __getitem__(self, key):
        batch_size, num_samples = self.weights.shape

        if isinstance(key, Union[int, slice]):
            # `batch_size = 1` and `num_samples = 1` is a common case,
            # avoids to select the batch through index `[0]` to slice.
            if batch_size * num_samples == 1:
                key = (0, 0, key)
            elif batch_size == 1 or num_samples == 1 and batch_size * num_samples != 1:
                raise IndexError(f"{key} should be a Tuple of size 2.")
            else:
                raise IndexError(f"{key} should be a Tuple of size 3.")

        if isinstance(key, Tuple):
            if batch_size == 1 and batch_size * num_samples != 1:
                if len(key) == 2:
                    key = (0, key[0], key[1])
                else:
                    raise IndexError(f"{key} should be a Tuple of size 2.")

            elif num_samples == 1 and batch_size * num_samples != 1:
                if len(key) == 2:
                    key = (key[0], 0, key[1])
                else:
                    raise IndexError(f"{key} should be a Tuple of size 2.")

            else:
                if len(key) != 3:
                    raise IndexError(f"{key} should be a Tuple of size 3.")

            batch_key, samples_key, ids_size_key = key

            # [NOTE] When dealing with a multidimensional List, the slicing doesn't propagate
            # to other dimensions. `tokenizer.decode(self.token_ids)` returns a list of strings.
            # Must use list comprehension to slice through dimensions.

            # [NOTE] The reason the indexing of `weights` depends on whether `batch_key` or `samples_key` are a slice or not,
            # is that indexing integer, removes a dimension in PyTorch.

            if isinstance(batch_key, slice):
                # Make sure that batch_key.start and batch_key.stop are defined.
                # Attributes `start` and `stop` are readonly,
                # can't do `batch_key.start = value`.
                batch_key = make_sure_slice_start_stop_defined(batch_key, 0, batch_size)
                if isinstance(samples_key, slice):
                    samples_key = make_sure_slice_start_stop_defined(
                        samples_key, 0, num_samples
                    )
                    sliced_prompts = [
                        prompt[ids_size_key]
                        for batch_key_idx in range(batch_key.start, batch_key.stop)
                        for prompt in self.tokenizer.decode(self.token_ids)[
                            batch_key_idx * num_samples : batch_key_idx * num_samples
                            + samples_key.stop
                        ]
                    ]
                    weights = self.weights[batch_key, samples_key]
                else:
                    sliced_prompts = [
                        self.tokenizer.decode(self.token_ids)[
                            batch_key_idx * num_samples + samples_key
                        ][ids_size_key]
                        for batch_key_idx in range(batch_key.start, batch_key.stop)
                    ]
                    weights = self.weights[batch_key, samples_key].unsqueeze(0)
            else:
                if isinstance(samples_key, slice):
                    samples_key = make_sure_slice_start_stop_defined(
                        samples_key, 0, num_samples
                    )
                    sliced_prompts = [
                        prompt[ids_size_key]
                        for prompt in self.tokenizer.decode(self.token_ids)[
                            batch_key * num_samples
                            + samples_key.start : batch_key * num_samples
                            + samples_key.stop
                        ]
                    ]
                    weights = self.weights[batch_key, samples_key].unsqueeze(0)
                else:
                    # `sliced_prompts` should always be a list, even with `1` element (reduces control flow).
                    sliced_prompts = [
                        self.tokenizer.decode(self.token_ids)[
                            batch_key * num_samples + samples_key
                        ][ids_size_key]
                    ]
                    weights = (
                        self.weights[batch_key, samples_key].unsqueeze(0).unsqueeze(0)
                    )

            # Uninteresting case.
            if isinstance(ids_size_key, int):
                return sliced_prompts
            if ids_size_key.start in (0, None):
                # [NOTE] Slices should be taken into account when their length is 0.
                # Conditions for the KV Cache to be saved when slicing `token_ids`:
                # (1) The `SequenceState` object has `batch_size == 1` and `num_samples == 1`.
                # (2) Any `SequenceState` sliced in a way such that `batch_size_after_slicing == 1` and `sample_size_after_slicing == 1`,
                # however user must modify the generator to have `num_samples == 1`. It has to match the `SequenceState`
                # or an exception `SampleMismatch` will be raised.
                batch_size_after_slicing, sample_size_after_slicing = weights.shape
                # Condition (1):
                if batch_size == 1 and num_samples == 1:
                    try:
                        (
                            ids_size_stop,
                            token_ids,
                        ) = token_level_slice_from_string_level_slice(
                            sliced_prompts[0], self.token_ids, self.tokenizer
                        )
                        attention_masks = build_attention_masks(
                            token_ids, self.tokenizer.pad_token_id
                        )
                        kv_cache = (
                            slice_while_rearranged_kv_cache(
                                rearrange_kv_cache(
                                    self.kv_cache, batch_size, num_samples
                                ),
                                ids_size_stop,
                                batch_key,
                                samples_key,
                            )
                            if self.kv_cache is not None
                            else None
                        )
                    except SlicingError:
                        warnings.warn(
                            "The current slicing does not preserve KV Cache, KV Cache will be recomputed in the next generation."
                        )
                        token_ids, attention_masks = self.tokenizer.encode(
                            sliced_prompts
                        )
                        kv_cache = None
                # Condition (2):
                # [NOTE] Users can select one element using slices [N:N+1], this is not taken into account.
                elif (
                    batch_size_after_slicing == 1 and sample_size_after_slicing == 1
                ) and (isinstance(batch_key, int) and isinstance(samples_key, int)):
                    try:
                        (
                            ids_size_stop,
                            token_ids,
                        ) = token_level_slice_from_string_level_slice(
                            sliced_prompts[0],
                            self.token_ids[
                                num_samples * batch_key + samples_key, :
                            ].unsqueeze(0),
                            self.tokenizer,
                        )
                        attention_masks = build_attention_masks(
                            token_ids, self.tokenizer.pad_token_id
                        )
                        kv_cache = (
                            slice_while_rearranged_kv_cache(
                                rearrange_kv_cache(
                                    self.kv_cache, batch_size, num_samples
                                ),
                                ids_size_stop,
                                batch_key,
                                samples_key,
                            )
                            if self.kv_cache is not None
                            else None
                        )
                    except SlicingError:
                        warnings.warn(
                            "The current slicing does not preserve KV Cache, KV Cache will be recomputed in the next generation."
                        )
                        token_ids, attention_masks = self.tokenizer.encode(
                            sliced_prompts
                        )
                        kv_cache = None
                else:
                    token_ids, attention_masks = self.tokenizer.encode(sliced_prompts)
                    kv_cache = None
                return SequenceState(
                    token_ids,
                    weights,
                    attention_masks,
                    # KV Cache should be rearranged back to its expanded version `batch_key*sample_key`.
                    expand_kv_cache(kv_cache),
                    self.tokenizer,
                )
            else:
                token_ids, attention_masks = self.tokenizer.encode(sliced_prompts)
                return SequenceState(
                    token_ids,
                    weights,
                    attention_masks,
                    None,
                    self.tokenizer,
                )

    def __str__(self):
        # [NOTE] Formatting to avoid a `SequenceState` object to be mistaken for a `List[str]`.
        # [NOTE] Special treatment to the common cases `num_samples*batch_size = 1` and `num_samples == 1 or batch_size == 1`.
        batch_size, num_samples = self.weights.shape
        if batch_size * num_samples == 1:
            return f"SequenceState{{'{self.tokenizer.decode(self.token_ids)[0]}'}}"
        elif num_samples == 1 or batch_size == 1:
            return f"SequenceState{{{self.tokenizer.decode(self.token_ids)}}}"
        else:
            formatted = ", ".join(
                [
                    "Samples(" + ", ".join([f"'{item}'" for item in seq]) + ")"
                    for seq in group_samples_into_sublists(
                        self.tokenizer.decode(self.token_ids), num_samples
                    )
                ]
            )
            return f"SequenceState{{{formatted}}}"

    def __iter__(self):
        return iter(self.tokenizer.decode(self.token_ids))

    def __add__(self, other):
        if isinstance(other, str) or islist(other, str):
            # Signal that KV cache + logprob need to be re-computed --> DONE.
            return token_level_add_sequence_state_to_str(self, other)
        if isinstance(other, SequenceState):
            # Concatenate token_ids -- DONE.
            # Concatenate logprobs -- Added the weights.
            # Signal that KV Cache after `other` needs to be recomputed' -- DONE but the
            # implementation was slower than resetting it, consequently it's not implemented.
            # see `complete_kv_cache_from_token_ids`.
            return token_level_add_sequence_state_to_sequence_state(self, other)


# Why?
# During generation within a batch, when an element finishes before the others,
# a right padding of `eos_token` tends to be added.
# So we basically end up with something like that:
"""
    [
    [pad_token, pad_token, token_1, token_2, generated_token_3, eos_token, eos_token],
    [token_1, token_2, token_3, token_4, generated_token_3, generated_token_4, eos_token],
    [pad_token, pad_token, token_1, generated_token_2, eos_token, eos_token, eos_token],
    ]
"""
# (1) We don't want to start the next generation from `eos_token`.
# (2) Makes it easier to concatenate two batches of `token_ids`
# since removing the padding becomes trivial (often `pad_token` = `eos_token`).

# How?
# We're going to (1) remove the last eos_tokens, then turn the other excess `eos_token`
# into `pad_token`.
"""
    [
    [pad_token, pad_token, eos_token_to_pad_token, new_pad_token, token_1, token_2, generated_token_3],
    [token_1, token_2, token_3, token_4, generated_token_3, generated_token_4],
    [pad_token, pad_token, eos_token_to_pad_token, eos_token_to_pad_token, token_1, generated_token_2],
    ]
"""
# [NOTE] Even if the prompt is the same and the answer is also the same, the generated tokens
# can be totally different.
# If the answer to some prompt is "Yes", the model could predict "Y" than "es"
# or directly "Yes".


def turn_right_eos_token_into_left_pad_token(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    eos_token: int,
):
    batch_size_mult_num_samples, _ = token_ids.shape

    # Removing last `eos_token`
    token_ids = token_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]
    kv_cache = slice_kv_cache(kv_cache, -1, slice(None, None, None))

    for batch_key_mult_num_samples_key in range(batch_size_mult_num_samples):
        if token_ids[batch_key_mult_num_samples_key, -1] == eos_token:
            token_ids[batch_key_mult_num_samples_key, :] = torch.roll(
                token_ids[batch_key_mult_num_samples_key, :], 1
            )
            attention_mask[batch_key_mult_num_samples_key, :] = torch.roll(
                attention_mask[batch_key_mult_num_samples_key, -1], 1
            )
            kv_cache = roll_kv_cache(kv_cache, batch_key_mult_num_samples_key)

    return token_ids, attention_mask, kv_cache


def remove_padding(batch_tensor: torch.Tensor, padding_token: int) -> List:
    lengths = (batch_tensor == padding_token).sum(dim=1)
    trimmed_batch = [
        sequence[length:] for sequence, length in zip(batch_tensor, lengths)
    ]
    return trimmed_batch


def add_padding(
    sequences: List[torch.tensor], padding_token, pad_on_right=False
) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)

    padded_sequences = torch.zeros(len(sequences), max_len).fill_(padding_token)
    for i, seq in enumerate(sequences):
        if pad_on_right:
            padded_sequences[i] = torch.nn.functional.pad(
                seq, (0, max_len - len(seq)), mode="constant", value=padding_token
            )
        else:
            padded_sequences[i] = torch.nn.functional.pad(
                seq, (max_len - len(seq), 0), mode="constant", value=padding_token
            )
    return padded_sequences


def build_attention_masks(
    token_ids_batch: torch.Tensor, padding_token: int
) -> torch.Tensor:
    attention_masks = (token_ids_batch != padding_token).float()
    return attention_masks


# TOKEN-LEVEL SequenceState --> str addition
def token_level_add_sequence_state_to_str(
    sequence_state: SequenceState, prompt: str
) -> SequenceState:
    prompt_to_token_ids, _ = sequence_state.tokenizer.encode(prompt)

    if prompt_to_token_ids.shape[0] != sequence_state.token_ids.shape[0]:
        raise BatchMismatchError(
            "A sequence and a string were added and their batch sizes were different."
        )

    # [POTENTIAL_BUG] Could `pad_token_id` not be defined in a tokenizer?
    # `pad_token_id` and `eos_token_id` are Optional in HuggingFace documentation, what to do then?
    pad_token_id = sequence_state.tokenizer.pad_token_id

    # [POTENTIAL_BUG] Removing the padding before concatenating,
    # [POTENTIAL_BUG] ASSUMING (it is for the tokenizer I'm using) the padding is on the left.
    token_ids = remove_padding(
        sequence_state.token_ids,
        pad_token_id,
    )
    prompt_to_token_ids = remove_padding(prompt_to_token_ids, pad_token_id)

    token_ids = [
        torch.cat((t1, t2), dim=0) for t1, t2 in zip(token_ids, prompt_to_token_ids)
    ]

    token_ids = add_padding(token_ids, pad_token_id).to(
        dtype=torch.long, device=sequence_state.token_ids.device
    )

    attention_masks = build_attention_masks(token_ids, pad_token_id).to(
        dtype=torch.long, device=sequence_state.attention_masks.device
    )

    weights = torch.zeros_like(
        sequence_state.weights,
        dtype=torch.float,
        device=sequence_state.weights.device,
    )

    warnings.warn("KV Cache will be recomputed in the next generation.")

    return SequenceState(
        token_ids, weights, attention_masks, None, sequence_state.tokenizer
    )


# TOKEN-LEVEL SequenceState --> SequenceState addition
def token_level_add_sequence_state_to_sequence_state(
    self_sequence_state: SequenceState, other_sequence_state: SequenceState
) -> SequenceState:
    if (
        self_sequence_state.token_ids.shape[0]
        != other_sequence_state.token_ids.shape[0]
    ):
        raise BatchMismatchError(
            "Sequences were added and their batch sizes were different."
        )

    # [POTENTIAL_BUG] Could `pad_token_id` not be defined in a tokenizer?
    # `pad_token_id` and `eos_token_id` are Optional in HuggingFace documentation, what to do then?
    pad_token_id = self_sequence_state.tokenizer.pad_token_id

    # [POTENTIAL_BUG] Removing the padding before concatenating,
    # [POTENTIAL_BUG] ASSUMING (it is for the tokenizer I'm using) the padding is on the left
    self_token_ids = remove_padding(
        self_sequence_state.token_ids,
        pad_token_id,
    )
    other_token_ids = remove_padding(other_sequence_state.token_ids, pad_token_id)

    concatenated_tensors = [
        torch.cat((t1, t2)) for t1, t2 in zip(self_token_ids, other_token_ids)
    ]

    token_ids = add_padding(concatenated_tensors, pad_token_id).to(
        dtype=torch.long, device=self_sequence_state.token_ids.device
    )

    attention_masks = build_attention_masks(token_ids, pad_token_id).to(
        dtype=torch.long, device=self_sequence_state.attention_masks.device
    )

    weights = self_sequence_state.weights + other_sequence_state.weights

    warnings.warn("KV Cache will be recomputed in the next generation.")

    return SequenceState(
        token_ids, weights, attention_masks, None, self_sequence_state.tokenizer
    )


def init_sequence_state(generator: SequenceGenerator, prompts, rng) -> SequenceState:
    if isinstance(prompts, str):
        prompts = [prompts]

    if rng is None:
        rng = torch.Generator(device=generator.device)
        rng.seed()

    prompt_token_ids, attention_masks = generator.tokenizer.encode(prompts)
    prompt_token_ids = prompt_token_ids.to(generator.device)
    attention_masks = attention_masks.to(generator.device)

    num_samples = generator.num_samples
    batch_size = len(prompts)

    prompt_token_ids = torch.repeat_interleave(prompt_token_ids, num_samples, dim=0)
    attention_masks = torch.repeat_interleave(attention_masks, num_samples, dim=0)

    # [NOTE] Small tweak to be able to get `batch_size` and `num_samples` from a SequenceState.
    weights = torch.zeros(
        batch_size, num_samples, dtype=torch.float, device=generator.device
    )
    return SequenceState(
        prompt_token_ids, weights, attention_masks, None, generator.tokenizer
    )


def get_next_sequence_state(
    generator: SequenceGenerator,
    sequence_state,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    rng: Optional[torch.Generator] = None,
) -> SequenceState:
    if isinstance(stop_at, str):
        stop_at = [stop_at]

    stop_sequences = stop_at

    batch_size, num_samples = sequence_state.weights.shape

    if num_samples != generator.num_samples:
        raise SampleMismatchError(
            f"Continuous generation can't proceed, Generator has a `num_samples == {generator.num_samples}` and SequenceState has a `num_samples == {num_samples}`. \
            A new generator with `num_samples == {num_samples}` should be utilized to proceed."
        )

    fsm_states = [0 for _ in range(batch_size * num_samples)]

    # Allows usage of different `generator` with different types of `fsm`.
    fsms = [generator.fsm.copy() for _ in range(batch_size * num_samples)]

    states = sequence_generator(
        generator.model,
        generator.sampler,
        fsms,
        sequence_state.token_ids,
        sequence_state.weights.view(-1),
        sequence_state.attention_masks,
        fsm_states,
        kv_cache=sequence_state.kv_cache,
        rng=rng,
    )

    while True:
        try:
            last_state = next(states)
            if max_tokens or stop_sequences:
                token_ids = last_state.token_ids
                generated_token_ids = generator.get_generated_token_ids(
                    sequence_state.token_ids, token_ids
                )
                if max_tokens and len(generated_token_ids[0]) >= max_tokens:
                    break
                if stop_sequences and generator.is_stop_sequence_found(
                    generator.tokenizer.decode(generated_token_ids), stop_sequences
                ):
                    break
        except StopIteration:
            break

    # Removes accumulated right `eos_token` and turns them into left `pad_token`.
    token_ids, attention_masks, kv_cache = turn_right_eos_token_into_left_pad_token(
        # Reason for using `copy`, RuntimeError: Inplace update to inference
        # tensor outside InferenceMode is not allowed.
        copy.copy(last_state.token_ids),
        copy.copy(last_state.attention_masks),
        copy.copy(last_state.kv_cache),
        sequence_state.tokenizer.eos_token_id,
    )

    return SequenceState(
        token_ids,
        last_state.weights.view(batch_size, num_samples),
        attention_masks,
        kv_cache,
        sequence_state.tokenizer,
    )


def continuous(generator: SequenceGenerator):
    def wrapper(
        prompt_or_sequence_state: Union[
            str, List[str], SequenceState, List[SequenceState]
        ],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        rng: Optional[torch.Generator] = None,
    ):
        if isinstance(prompt_or_sequence_state, str) or islist(
            prompt_or_sequence_state, str
        ):
            next_sequence_state = init_sequence_state(
                generator, prompt_or_sequence_state, rng
            )
            return get_next_sequence_state(
                generator, next_sequence_state, max_tokens, stop_at, rng
            )
        elif isinstance(prompt_or_sequence_state, SequenceState) or islist(
            prompt_or_sequence_state, SequenceState
        ):
            return get_next_sequence_state(
                generator, prompt_or_sequence_state, max_tokens, stop_at, rng
            )
        else:
            raise TypeError(
                "Invalid input type, input should be str, List[str], SequenceState or List[SequenceState]."
            )

    return wrapper
