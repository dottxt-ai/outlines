"""Guides to control generation in steerable models.

Logits processors rely on guides to control the generation process.
"""

from typing import TYPE_CHECKING, Any, Union

from outlines_core.fsm.guide import Generate
from outlines_core.fsm.guide import Guide as CoreGuide
from outlines_core.fsm.guide import RegexGuide as CoreRegexGuide
from outlines_core.fsm.guide import Write
from outlines_core.fsm.guide import (
    create_states_mapping as uncached_create_states_mapping,
)

from outlines.caching import cache

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


Instruction = Union[Write, Generate]


class Guide(CoreGuide):
    """Base definition of a generation guide.

    A generation guide defines the behavior of a finite-state machine that
    guides a text generation procedure. Unlike the DFAs built from regular
    expressions guides, it can also emit a `Write` instructions which tells
    the model that it can append a sequence of tokens (or token word) instead
    of generating it.

    """
    initial_state: Any


class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""
    final_state = 1
    initial_state = 0

    def __init__(self, tokenizer: "Tokenizer"):
        """
        Parameters
        ----------
        tokenizer
            The tokenizer used to convert tokens to ids.

        """
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary.values()

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction.

        Parameters
        ----------
        state
            The guide's current state.

        Returns
        -------
        Instruction
            An `Instruction` instance.

        """
        if self.is_final_state(state):
            return Write([self.eos_token_id])
        return Generate(None)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Return the next state.

        Parameters
        ----------
        state
            The guide's current state.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        int
            The next state.

        """
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state

        return self.initial_state

    def is_final_state(self, state: int) -> bool:
        """Return whether the given state is a final state.

        Parameters
        ----------
        state
            The guide's current state.

        Returns
        -------
        bool
            Whether the given state is a final state.

        """
        return state == self.final_state

    def copy(self) -> "StopAtEOSGuide":
        """Return itself as there is no need to copy."""
        return self


@cache()
def cached_create_states_mapping(regex_string, tokenizer, *args, **kwargs):
    """Wrap the uncached create_states_mapping function in a cache."""
    return uncached_create_states_mapping(
        regex_string, tokenizer, *args, **kwargs
    )


class RegexGuide(CoreRegexGuide):
    """Guide to generate text in the language of a regular expression.

    This class is a wrapper around the CoreRegexGuide class that adds a cache
    to the create_states_mapping function.

    """

    @classmethod
    def from_regex(
        cls,
        regex_string: str,
        tokenizer,
        **kwargs,
    ):
        """Create a RegexGuide from a regular expression.

        Parameters
        ----------
        regex_string
            The regular expression to generate text from.
        tokenizer
            The tokenizer to use to convert tokens to ids.
        kwargs
            Additional keyword arguments to pass to the CoreRegexGuide constructor.

        Returns
        -------
        RegexGuide
            A RegexGuide instance.

        """
        return super().from_regex(
            regex_string,
            tokenizer,
            _create_states_mapping=cached_create_states_mapping,
            **kwargs,
        )
