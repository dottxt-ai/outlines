import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, Union,Any

import mlx.core as mx
import numpy as np 

from outlines.fsm.fsm import FSMState
from time import time
if TYPE_CHECKING:
    from outlines.fsm.fsm import FSM
    from outlines.generate.samplers_mlx import Sampler_mlx
    from outlines.models.tokenizer import Tokenizer


@dataclasses.dataclass(frozen=True)
class GenerationState:
    token_ids: mx.array
    kv_cache: mx.array
    logits: mx.array
    fsm_states: List[FSMState]


def init_generator_state(
    tokenizer: "Tokenizer",
    prompt: Union[str, List[str]],
    kv_cache: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Initialize the generation state.

    This method is responsible for encoding the prompt.

    Parameters
    ----------
    prompt
        The prompt on which the generation is conditioned.
    
    Returns
    -------
    A `GenerationState` object.

    """
    token_ids, attention_masks = tokenizer.encode(prompt)

    return token_ids, attention_masks, kv_cache


def sequence_generator(
    token_generator: Callable,
    fsm: "FSM",
    init_state: Tuple[Any,Any,Any],
    fsm_states: List[FSMState],
    rng: mx.array
) -> Iterator[GenerationState]:
    """Generates sequences of tokens.

    Parameters
    ----------
    token_generator
        A callable that generate a new token given the current generation state
        and logits biases.
    fsm
        The finite-state machine that drives the text generation.
    init_state
        The initial generation state for the batches.
    fsm_states
        The initial states of the finite-state machine for each sequence in the batch.
    rng
        The state of the random number generator.
    Yields
    ------
    A new sequence.

    """
    token_ids, attention_masks, kv_cache = init_state
    while True:
        allowed_tokens = get_allowed_tokens(fsm, fsm_states)

        next_token_ids, kv_cache, logits, _ = token_generator(
            token_ids,
            attention_masks,
            kv_cache,
            rng=rng,
            allowed_tokens=allowed_tokens,
        )

        token_ids = update_token_ids(token_ids, next_token_ids)
        attention_masks = expand_attention_masks(attention_masks)

        fsm_states = get_next_fsm_states(fsm, fsm_states, next_token_ids)
        is_finished = is_generation_finished(fsm, fsm_states)

        if is_finished:
            yield GenerationState(token_ids, kv_cache, logits, fsm_states)
            return

        yield GenerationState(token_ids, kv_cache, logits, fsm_states)


def token_generator(model, sampler: "Sampler_mlx") -> Callable:
    """Generate one token at a time.

    This process is designed to be steered by another supervising
    process that supplies the current sequence and the indices
    of the tokens to mask before sampling.

    Parameters
    ----------
    model
        A model that takes a sequence of tokens as an input and
        returns a probability distribution over the next tokens.
    sampler
        A function that samples tokens from a probability
        distribution over the next tokens.

    Returns
    -------
    A tuple that contains a tensor with the sampled tokens, a tensor with
    the K-V cache for the sequence and the tensor that contains the next-token
    logits that were returned by the model.

    """
    def generate(
        token_ids: mx.array,
        attention_masks: mx.array,
        kv_cache: mx.array,
        allowed_tokens: List[List[int]],
        rng: mx.array,
    ) -> Union[mx.array, mx.array, mx.array, mx.array]:
        try:
            logits, new_kv_cache = model(token_ids, attention_masks, kv_cache)
        except IndexError:  # Exceeding the context length
            raise IndexError(
                "The input length exceeds the context length of the model."
            )

        biased_logits = bias_logits(logits, allowed_tokens)
        #sampler to mlx
        next_token_ids = sampler(biased_logits,1, rng)

        return next_token_ids, new_kv_cache, logits, biased_logits

    return generate


def get_next_fsm_states(
    fsm: "FSM", fsm_states: List[FSMState], next_token_ids: mx.array
) -> List[FSMState]:
    """

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    next_token_ids
        The tokens that were just generated.

    Returns
    -------
    A `mx.array` object that represents the next logit mask.

    """
    return [
        fsm.next_state(fsm_state, int(np.array(token_id[0])), idx)
        for idx, fsm_state, token_id in zip(
            range(len(fsm_states)), fsm_states, next_token_ids
        )
    ]


def get_allowed_tokens(fsm: "FSM", fsm_states: List[FSMState]) -> List[List[int]]:
    """Get the new instructions for each sequence from the finite-state machine.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    A nested list that contains the ids of the logits to keep.

    """
    return [fsm.allowed_token_ids(state, idx) for idx, state in enumerate(fsm_states)]


def is_generation_finished(fsm: "FSM", fsm_states: List[FSMState]) -> bool:
    """Determine if the generation is finished.

    A generation is considered finished if the FSM of every sequence in the
    batch is in a final state.

    A better solution is to return finished sequences as soon as their FSM
    is in a final state.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    Whether all sequences are finished sampling.

    """
    return all([fsm.is_final_state(state, idx) for idx, state in enumerate(fsm_states)])


def update_token_ids(
    token_ids: mx.array, next_token_ids: mx.array
) -> mx.array:
    """Append the sampled tokens to the running sequence of tokens.

    Parameters
    ----------
    token_ids
        The current token sequences
    next_token_ids
        The tokens that were just generated and that we need to append
        to the existing sequences.

    Returns
    -------
    A new sequence of token ids that contains the tokens that were
    just generated.

    """
    updated_token_ids = mx.concatenate([token_ids, next_token_ids], axis=-1)
    return updated_token_ids


def expand_attention_masks(attention_masks: mx.array) -> mx.array:
    """Expand the attention masks.

    Parameters
    ----------
    attention_masks
        The attention masks for each sequence in the batch.

    Returns
    -------
    The attention masks padded with 1s.

    """
    expanded_mask = mx.concatenate(
        [
            attention_masks,
            mx.ones(
                attention_masks.shape[:-1] + [1]
            ),
        ],
        axis=-1,
    )
    return expanded_mask


def bias_logits(
    logits: mx.array,
    ids_to_mask: List[List[int]],
) -> mx.array:
    """Mask the logits.

    The function iterates over a nested list where each list corresponds to the
    indices that need to be masked for each row in the array.

    Parameters
    ----------
    logits
        Two dimensional tensor that contains the next-token probability
        distribution.
    ids_to_mask
        The ids to mask in each dimension.

    Returns
    -------
    A view of the original logits tensor where some values are masked.

    """
    biased_logits = mx.zeros(logits.shape)
    for i, ids in enumerate(mx.array(ids_to_mask)):
        mask = mx.full((logits.shape[-1],), -math.inf)
        mask[ids] = 0
        biased_logits[i] = logits[i] + mask
    return biased_logits
