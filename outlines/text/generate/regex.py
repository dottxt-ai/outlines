import math
from json import dumps
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import interegular
import torch
from pydantic import BaseModel

from outlines.text.fsm import create_fsm_index_tokenizer, make_deterministic_fsm
from outlines.text.generate.continuation import Continuation
from outlines.text.json_schema import build_regex_from_schema

if TYPE_CHECKING:
    from outlines.text.generate.sample import Sampler


class Regex(Continuation):
    """Represents a regex-based generation model.

    `Regex` instances are constrained generation models that only generate
    sequences matching a given regex.

    >>> import outlines.text as text
    >>> generator = text.generate.regex(model, "(0|[1-9][0-9]+)")

    Sequences can then be generated from a prompt as follows:

    >>> sequence_1 = generator("Return an integer between 0 and 10")
    >>> sequence_2 = generator("Rate the movie "Hackers" on a scale from 0 to 10")

    .. note:
        Reuse instances of these guided generators (e.g. `generator` from the
        above example) whenever possible, because constructing them has more
        overhead than generating token sequences from them.

    """

    def __init__(
        self,
        model,
        regex_string: str,
        max_tokens: Optional[int] = None,
        *,
        sampler: Optional["Sampler"] = None,
        stop: Union[str, List[str]] = [],
        allow_empty_tokens: bool = True,
        initial_state: Optional[int] = None,
        final_states: Optional[Set[int]] = None,
        states_to_token_maps: Optional[Dict[int, Dict[int, int]]] = None,
        empty_token_ids: Optional[Set[int]] = None,
    ):
        """

        Parameters
        ----------
        model
            The instance of the model used to generate next-token probabilities.
        regex_string
            The regex with which the token sampling process is guided/constrained.
        max_tokens
            The maximum number of tokens to be sampled.
        sampler
            The function used to draw samples.  Defaults to
            `outlines.text.generate.sample.multinomial`.  See
            `outlines.text.generate.sample.Sampler` for the expected form of
            such functions.
        stop
            Optional stopping string(s).
        allow_empty_tokens
            Allow sampling of tokens corresponding to empty strings.
        states_to_token_maps
            Pre-computed map of FSM start states to maps between token ids and their
            corresponding FSM end states.
        empty_token_ids
            Pre-computed set of token ids for tokens that are empty strings.

        """
        super().__init__(model, max_tokens, sampler, stop)

        if (
            states_to_token_maps is None
            or empty_token_ids is None
            or initial_state is None
            or final_states is None
        ):
            regex_pattern = interegular.parse_pattern(regex_string)
            regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

            (
                self.states_to_token_maps,
                self.empty_token_ids,
            ) = create_fsm_index_tokenizer(regex_fsm, model.tokenizer)
            self.initial_state = regex_fsm.initial
            self.final_states = regex_fsm.finals
        else:
            self.initial_state = initial_state
            self.final_states = final_states
            self.states_to_token_maps = states_to_token_maps
            self.empty_token_ids = empty_token_ids

        # Check whether a terminal path (from the initial state of the FSM to
        # one of its terminal states) exists, raise an exception otherwise.
        if not any(
            self.final_states.intersection(v.values())
            for v in self.states_to_token_maps.values()
        ):
            raise ValueError(
                "The vocabulary does not allow us to build a sequence that matches the input regex"
            )

        # When an EOS is observed, the last FSM state becomes `-1`.
        self.last_fsm_states: List[int] = []
        self.mask_cache: Dict[Tuple[int, int], torch.LongTensor] = {}
        self.regex_string = regex_string
        self.allow_empty_tokens = allow_empty_tokens

    def create_proposal(
        self, generated_token_ids: torch.LongTensor, logits: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Modify the next-token logits so that only valid tokens can be generated.

        Parameters
        ----------
        generated_token_ids
            The token ids generated so far.
        logits
            The next-token logits.

        """

        assert generated_token_ids.ndim == 2

        if len(self.last_fsm_states) == 0:
            self.last_fsm_states = [self.initial_state for _ in range(logits.shape[0])]

        masks = []

        for i, (token_seq, last_state) in enumerate(
            zip(
                generated_token_ids,
                self.last_fsm_states,
            )
        ):
            if token_seq.shape[0] > 0:
                # Get the last token that was sampled
                last_token = int(token_seq[-1])

                if last_token in self.empty_token_ids:
                    # An empty token was sampled, so the FSM state hasn't changed
                    next_state = last_state
                    next_token_ids = list(self.states_to_token_maps[last_state].keys())

                elif last_token != self.model.tokenizer.eos_token_id:
                    # If we previously ended with an EOS, we shouldn't be
                    # getting/sampling any more non-EOS tokens.
                    assert last_state > -1

                    last_token_to_end_state = self.states_to_token_maps[last_state]

                    next_state = last_token_to_end_state[last_token]

                    next_tokens_to_end_states = self.states_to_token_maps.get(
                        next_state
                    )

                    if next_tokens_to_end_states is None:
                        # If there are no transitions from the current state,
                        # then we must've been in a final state of the FSM.
                        # We produce EOS tokens from here on.
                        assert next_state in self.final_states
                        next_state = -1
                        next_token_ids = [self.model.tokenizer.eos_token_id]
                    else:
                        next_token_ids = list(next_tokens_to_end_states.keys())
                else:
                    # Since we already have an EOS, only sample EOS tokes from
                    # here on.
                    next_state = -1
                    next_token_ids = [self.model.tokenizer.eos_token_id]
            else:
                # There weren't any previous tokens, so we can't update the state
                next_state = last_state
                next_token_ids = list(self.states_to_token_maps[last_state].keys())

            mask = self._get_mask_for_state(
                next_state, logits.shape[-1], next_token_ids
            )
            masks.append(mask)
            self.last_fsm_states[i] = next_state

        mask = torch.concatenate(masks, dim=0)

        return logits + mask

    def _get_mask_for_state(
        self, state: int, size: int, next_token_ids: List[int]
    ) -> torch.LongTensor:
        mask = self.mask_cache.get((state, size))

        if mask is None:
            mask = torch.full(
                (size,),
                -math.inf,
                device=self.device,
            )

            if self.allow_empty_tokens:
                token_ids = list(self.empty_token_ids) + next_token_ids
            else:
                token_ids = next_token_ids

            mask[token_ids] = 0
            mask = mask.unsqueeze(0)
            self.mask_cache[(state, size)] = mask

        return mask

    def postprocess_completions(self, completions: List[str]) -> List[str]:
        self.last_fsm_states.clear()
        return super().postprocess_completions(completions)


def regex(
    model,
    regex_string: str,
    max_tokens: Optional[int] = None,
    *,
    sampler: Optional["Sampler"] = None,
    allow_empty_tokens: bool = True,
):
    """Generate text sequences that match the input regex.

    .. note:
        Reuse instances of these guided generators whenever possible,
        because constructing them has more overhead than generating
        token sequences from them.  See the docstring for `Regex`.

    Parameters
    ----------
    model
        The language model to use to compute the next-token logits.
    regex_string
        The regular expression that generated expressions must match.
    max_tokens
        The maximum number of tokens to generate.
    sampler
        The function used to draw samples.  Defaults to
        `outlines.text.generate.sample.multinomial`.  See
        `outlines.text.generate.sample.Sampler` for the expected form of
        such functions.
    allow_empty_tokens
        Allow sampling of tokens corresponding to empty strings.

    """
    return Regex(
        model,
        regex_string,
        max_tokens,
        sampler=sampler,
        allow_empty_tokens=allow_empty_tokens,
    )


def integer(
    model,
    max_tokens: Optional[int] = None,
    *,
    sampler: Optional["Sampler"] = None,
    allow_empty_tokens: bool = True,
):
    """Generate integers.

    The regex used to constrain the generation optionally matches plus or minus
    signs and forbids leading zeros (even if the `int` function in Python allows
    them).

    .. note:
        Reuse instances of these guided generators whenever possible,
        because constructing them has more overhead than generating
        token sequences from them.  See the docstring for `Regex`.

    Parameters
    ----------
    model
        The language model to use to compute the next-token logits.
    max_tokens
        The maximum number of tokens to generate.
    sampler
        The function used to draw samples.  Defaults to
        `outlines.text.generate.sample.multinomial`.  See
        `outlines.text.generate.sample.Sampler` for the expected form of
        such functions.
    allow_empty_tokens
        Allow sampling of tokens corresponding to empty strings.

    """
    return Regex(
        model,
        r"[-+]?\d+",
        max_tokens,
        sampler=sampler,
        allow_empty_tokens=allow_empty_tokens,
    )


def float(
    model,
    max_tokens: Optional[int] = None,
    *,
    sampler: Optional["Sampler"] = None,
    allow_empty_tokens: bool = True,
):
    """Generate floating-point numbers.

    The regex used to constrain the generation optionally matches plus or minus
    signs, and forbids leading zeros (even if the `float` function in Python
    allows them).

    .. note:
        Reuse instances of these guided generators whenever possible,
        because constructing them has more overhead than generating
        token sequences from them.  See the docstring for `Regex`.

    Parameters
    ----------
    model
        The language model to use to compute the next-token logits.
    max_tokens
        The maximum number of tokens to generate.
    sampler
        The function used to draw samples.  Defaults to
        `outlines.text.generate.sample.multinomial`.  See
        `outlines.text.generate.sample.Sampler` for the expected form of
        such functions.
    allow_empty_tokens
        Allow sampling of tokens corresponding to empty strings.

    """
    return Regex(
        model,
        r"([+-]?((0|[1-9]+)([.][0-9]*)?)|([.][0-9]+))",
        max_tokens,
        sampler=sampler,
        allow_empty_tokens=allow_empty_tokens,
    )


def choice(
    model,
    choices: List[str],
    max_tokens: Optional[int] = None,
    *,
    sampler: Optional["Sampler"] = None,
    allow_empty_tokens: bool = True,
):
    """Choose between different sequences.

    .. note:
        Reuse instances of these guided generators whenever possible,
        because constructing them has more overhead than generating
        token sequences from them.  See the docstring for `Regex`.

    Parameters
    ----------
    model
        The language model to use to compute the next-token logits.
    max_tokens
        The maximum number of tokens to generate.
    sampler
        The function used to draw samples.  Defaults to
        `outlines.text.generate.sample.multinomial`.  See
        `outlines.text.generate.sample.Sampler` for the expected form of
        such functions.
    allow_empty_tokens
        Allow sampling of tokens corresponding to empty strings.
    """
    regex_str = r"(" + r"|".join(choices) + r")"
    return Regex(
        model,
        regex_str,
        max_tokens,
        sampler=sampler,
        allow_empty_tokens=allow_empty_tokens,
    )


def json(
    model,
    schema: Union[str, BaseModel],
    max_tokens: Optional[int] = None,
    *,
    sampler: Optional["Sampler"] = None,
    allow_empty_tokens: bool = True,
):
    """Generate a text sequence that follows a JSON schema or Pydantic model.

    .. note:
        Reuse instances of these guided generators whenever possible,
        because constructing them has more overhead than generating
        token sequences from them.  See the docstring for `Regex`.

    Parameters
    ---------
    model
        The language model to use to compute the next-token logits.
    schema
        The JSON schema or Pydantic model that guides the generation.
    max_tokens
        The maximum number of tokens to generate.
    sampler
        The function used to draw samples.  Defaults to
        `outlines.text.generate.sample.multinomial`.  See
        `outlines.text.generate.sample.Sampler` for the expected form of
        such functions.
    allow_empty_tokens
        Allow sampling of tokens corresponding to empty strings.

    """
    if isinstance(schema, type(BaseModel)):
        schema = dumps(schema.model_json_schema())

    regex_str = build_regex_from_schema(schema)

    return Regex(
        model,
        regex_str,
        max_tokens,
        sampler=sampler,
        allow_empty_tokens=allow_empty_tokens,
    )
