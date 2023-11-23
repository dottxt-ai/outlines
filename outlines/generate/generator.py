import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, Union

import torch

from outlines.index.index import FSMState

if TYPE_CHECKING:
    from outlines.generate.samplers import Sampler
    from outlines.index.index import FSM


@dataclasses.dataclass(frozen=True)
class GenerationState:
    token_ids: torch.Tensor
    kv_cache: torch.Tensor
    logits: torch.Tensor


class SequenceGenerator:
    def __init__(self, fsm, model, sampler, device):
        self.generate_token = token_generator(model, sampler)
        self.fsm = fsm
        self.tokenizer = model.tokenizer
        self.device = device

    def init_generation_state(
        self,
        prompt: Union[str, List[str]],
        kv_cache: Optional[torch.Tensor] = None,
    ):
        """Initialize the generation state.

        This method is responsible for encoding the prompt, moving token ids
        to the device and initializing the random number generator.

        Parameters
        ----------
        prompt
            The prompt on which the generation is conditioned.
        rng
            The state of the random number generator.

        Returns
        -------
        A `GenerationState` object.

        """
        token_ids, attention_masks = self.tokenizer.encode(prompt)
        token_ids = token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        return token_ids, attention_masks, kv_cache

    def __call__(
        self,
        prompt,
        kv_cache: Optional[torch.tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Union[str, List[str]]:
        sequence_generator = self.stream(prompt, kv_cache, rng)
        tokens = [token for token in sequence_generator]
        sequences = ["".join(sequence) for sequence in list(zip(*tokens))]
        return sequences if len(sequences) > 1 else sequences[0]

    def stream(
        self,
        prompt: str,
        kv_cache: Optional[torch.tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> Iterator[Union[List[str], str]]:
        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        init_state = self.init_generation_state(prompt, kv_cache)

        token_ids = init_state[1]
        num_sequences = token_ids.shape[0]

        init_fsm_states = [FSMState(0) for _ in range(num_sequences)]

        states = sequence_generator(
            self.generate_token, self.fsm, init_state, init_fsm_states, rng
        )

        def token_generator() -> Iterator[Union[List[str], str]]:
            while True:
                try:
                    sequence = next(states)
                except StopIteration:
                    return

                next_token_ids = sequence.token_ids[:, -1]
                next_tokens = self.tokenizer.decode(next_token_ids)

                yield next_tokens

        return token_generator()


def sequence_generator(
    token_generator: Callable,
    fsm: "FSM",
    init_state: Tuple,
    fsm_states: List[FSMState],
    rng: torch.Generator,
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

    Yields
    ------
    A new sequence.

    """
    token_ids, attention_masks, kv_cache = init_state
    while True:
        logits_masks = get_next_instruction(fsm, fsm_states)

        next_token_ids, kv_cache, logits = token_generator(
            token_ids,
            attention_masks,
            kv_cache,
            rng=rng,
            logits_masks=logits_masks,
        )

        token_ids = update_token_ids(token_ids, next_token_ids)
        attention_masks = expand_attention_masks(attention_masks)

        fsm_states = get_next_fsm_states(fsm, fsm_states, next_token_ids)
        is_finished = is_generation_finished(fsm, fsm_states)

        if is_finished:
            yield GenerationState(token_ids, kv_cache, logits)
            return

        yield GenerationState(token_ids, kv_cache, logits)


@torch.inference_mode
def token_generator(model, sampler: "Sampler") -> Callable:
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
        token_ids,
        attention_masks,
        kv_cache,
        logits_masks,
        rng: torch.Generator,
    ) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            logits, new_kv_cache = model(token_ids, attention_masks, kv_cache)
        except IndexError:  # Exceeding the context length
            raise IndexError(
                "The input length exceeds the context length of the model."
            )

        biased_logits = bias_logits(logits, logits_masks)
        next_token_ids = sampler(biased_logits, 1, rng)

        return next_token_ids, new_kv_cache, biased_logits

    return generate


def get_next_fsm_states(
    fsm: "FSM", fsm_states: List[FSMState], next_token_ids: torch.Tensor
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
    A `torch.Tensor` object that represents the next logit mask.

    """
    return [
        fsm.next_state(fsm_state, int(token_id[0]))
        for fsm_state, token_id in zip(fsm_states, next_token_ids)
    ]


def get_next_instruction(fsm: "FSM", fsm_states: List[FSMState]) -> torch.Tensor:
    """Get the new instructions for each sequence from the finite-state machine.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    A nested list that contains the ids of the logits to bias.

    """
    return [fsm.next_instruction(state) for state in fsm_states]


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
    return all([fsm.is_final_state(state) for state in fsm_states])


def update_token_ids(
    token_ids: torch.Tensor, next_token_ids: torch.Tensor
) -> torch.Tensor:
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
    return torch.concatenate([token_ids, next_token_ids], dim=-1)


def expand_attention_masks(attention_masks: torch.Tensor) -> torch.Tensor:
    """Expand the attention masks.

    Parameters
    ----------
    attention_masks
        The attention masks for each sequence in the batch.

    Returns
    -------
    The attention masks padded with 1s.

    """
    return torch.concatenate(
        [
            attention_masks,
            torch.ones(
                attention_masks.shape[:-1] + (1,), device=attention_masks.device
            ),
        ],
        axis=-1,
    )


<<<<<<< HEAD
@torch.inference_mode()
=======
def update_logprobs(logprobs, next_token_ids, next_token_logits):
    """Update the sequences' total logprob.

    Parameters
    ----------
    logprobs
        The current log-probabilities for each sequence.
    next_token_ids
        The token ids that were just sampled
    next_token_logits
        The logits returned by the model.

    """
    next_token_logprobs = torch.nn.LogSoftmax(dim=-1)(next_token_logits)
    new_logprobs = next_token_logprobs[
        range(next_token_ids.shape[0]), next_token_ids.flatten()
    ]
    return logprobs + new_logprobs

@torch.inference_mode
def bias_logits(
    logits: torch.Tensor,
    ids_to_mask: List,
) -> torch.Tensor:
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
    for i, ids in enumerate(ids_to_mask):
        logits[i, ids] = -math.inf
    return logits
