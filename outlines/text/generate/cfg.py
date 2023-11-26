import math
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from lark import Lark, UnexpectedCharacters, UnexpectedToken

from outlines.text.generate.continuation import Continuation
from outlines.text.generate.regex import regex

if TYPE_CHECKING:
    from outlines.text.generate.regex import Regex
    from outlines.text.generate.sample import Sampler


class CFG(Continuation):
    """Represents a cfg-based generation model.

    `CFG` instances are constrained generation models that only generate
    sequences matching a given context-free grammar.

    >>> import outlines.text as text
    >>> grammar = '''
    ...     start: palindrome
    ...     palindrome: letter
    ...         | "a" palindrome "a"
    ...         | "b" palindrome "b"
    ...         | "c" palindrome "c"
    ...     letter: "a" | "b" | "c"
    ... '''
    >>> generator = text.generate.cfg(model, grammar)

    Sequences can then be generated from a prompt as follows:

    >>> sequence = generator("Return a palindrome using only the characters 'a', 'b', and 'c': ")

    .. note:
        Reuse instances of these guided generators (e.g. `generator` from the
        above example) whenever possible, because constructing them has more
        overhead than generating token sequences from them.

    """

    def __init__(
        self,
        model,
        cfg_string: str,
        max_tokens: Optional[int] = None,
        *,
        sampler: Optional["Sampler"] = None,
        stop: Union[str, List[str]] = [],
        allow_empty_tokens: bool = True,
    ):
        """

        Parameters
        ----------
        model
            The instance of the model used to generate next-token probabilities.
        cfg_string
            The cfg with which the token sampling process is guided/constrained.
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

        """
        super().__init__(model, max_tokens, sampler, stop)
        self.allow_empty_tokens = allow_empty_tokens

        self.parser = Lark(
            cfg_string,
            parser="lalr",
            lexer="contextual",
            propagate_positions=False,
            maybe_placeholders=False,
            regex=True,
        )
        self.terminal_regexps = dict()
        for terminal in self.parser.terminals:
            if terminal.pattern is not None:
                self.terminal_regexps[terminal.name] = terminal.pattern.to_regexp()
        self.terminal_regexps["$END"] = self.model.tokenizer.eos_token

        self._last_tokens: torch.LongTensor
        self._completions: List[str]
        self._regexps: List["Regex"]
        self._starts: List[int]
        self._increments: List[int]
        self._stop_flags: List[bool]
        self._eos_on_flags: List[bool]
        self._eos_off_flags: List[bool]

    def _set_states(self, batch_size: int) -> None:
        self._completions = [""] * batch_size
        self._regexps = [None] * batch_size  # type: ignore
        self._starts = [0] * batch_size
        self._increments = [0] * batch_size
        self._stop_flags = [False] * batch_size
        self._eos_on_flags = [False] * batch_size
        self._eos_off_flags = [False] * batch_size

    def _keep_indices(self, keep: List[int]) -> None:
        update = lambda x: [e for i, e in enumerate(x) if i in keep]
        self._completions = update(self._completions)
        self._regexps = update(self._regexps)
        self._starts = update(self._starts)
        self._increments = update(self._increments)
        self._stop_flags = update(self._stop_flags)
        self._eos_on_flags = update(self._eos_on_flags)
        self._eos_off_flags = update(self._eos_off_flags)

    def _filter_indices(self, current_tokens: torch.LongTensor) -> None:
        keep = []
        for x in current_tokens:
            matches = [
                i for i, p in enumerate(self._last_tokens) if (p == x[: len(p)]).all()
            ]
            for m in matches:
                if m not in keep:
                    keep.append(m)
                    break
        self._keep_indices(keep)

    def _get_next_proposal(
        self,
        input_str: str,
        token_ids: torch.LongTensor,
        logits: torch.DoubleTensor,
        idx: int,
    ) -> torch.DoubleTensor:
        self._eos_on_flags[idx] = False
        self._eos_off_flags[idx] = False

        interactive = self.parser.parse_interactive(input_str)

        try:
            interactive.exhaust_lexer()  # if the regex is incomplete, this will raise an exception
            # in that case, we will just continue with the current regex
            # else, we will assess whether to update to the next regex

            options = {self.terminal_regexps[x] for x in interactive.accepts()}

            if self.terminal_regexps["$END"] in options:
                # if eos is a valid continuation from the cfg parser, find regex mask without it then add back later
                # this is because we can't pass `<|endoftext|>` etc. to the current regex implementation
                options.remove(self.terminal_regexps["$END"])
                self._eos_on_flags[idx] = True
                if len(options) == 0:
                    # if eos is the only valid continuation, stop generating past that point
                    self._stop_flags[idx] = True
                    return None  # type: ignore

            # assess whether to build new regex or continue with current one
            subcompletion = "".join(
                self.model.tokenizer.decode(token_ids[0, self._starts[idx] :])
            )
            # if there is a regex in progress that might be continued from
            if subcompletion != "":
                # check what would be proposed next
                regex_proposal = self._regexps[idx].create_proposal(
                    token_ids[:, self._starts[idx] :], logits
                )
                rng = torch.Generator(device=self.device)
                rng.seed()
                # sample from that proposal
                sample = self.sampler(regex_proposal, 1, rng).item()
                # as long as sample is not eos, keep using the current regex
                if sample != self.model.tokenizer.eos_token_id:
                    # but make sure to remove eos from proposed mask
                    self._eos_off_flags[idx] = True
                    return regex_proposal
            # else move to next regex
            regex_str = r"(" + r"|".join([r"(" + x + r")" for x in options]) + r")"
            self._regexps[idx] = regex(
                self.model,
                regex_str,
                max_tokens=self.max_tokens,
                sampler=self.sampler,
                allow_empty_tokens=self.allow_empty_tokens,
            )
            # update start index to ignore tokens generated prior to this regex
            self._starts[idx] += self._increments[idx]
            # and reset the number of tokens generated from this regex
            self._increments[idx] = 0

        except (UnexpectedCharacters, UnexpectedToken):
            pass  # keep using the current regex

        return self._regexps[idx].create_proposal(
            token_ids[:, self._starts[idx] :], logits
        )

    def _get_masked_logits(
        self, token_ids: torch.LongTensor, logits: torch.DoubleTensor, idx: int
    ) -> torch.DoubleTensor:
        def add_eos(masked: torch.DoubleTensor) -> torch.DoubleTensor:
            original = logits.clone().flatten()[self.model.tokenizer.eos_token_id]
            masked[:, self.model.tokenizer.eos_token_id] = original
            return masked

        # prepare the next regex proposal and set relevant flags
        masked = self._get_next_proposal(self._completions[idx], token_ids, logits, idx)
        self._increments[idx] += 1  # track tokens generated for current regex

        # if nothing left to generate, mask all except eos
        if self._stop_flags[idx]:
            masked = add_eos(logits.clone() - math.inf)

        # else if current regex is continuing, make sure can't generate eos
        elif self._eos_off_flags[idx]:
            masked[:, self.model.tokenizer.eos_token_id] = -math.inf

        # else if current regex not continuing and eos proposed by cfg parser, keep the original eos logits
        elif self._eos_on_flags[idx]:
            masked = add_eos(masked)

        return masked

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

        if generated_token_ids.shape[1] == 0:  # no tokens generated yet
            # initialize progress tracking for each batch element
            self._set_states(logits.shape[0])
        else:
            # if batch size has changed due to individual completions
            if len(generated_token_ids) != len(self._last_tokens):
                # filter out batch elements that have been completed to maintain alignment
                self._filter_indices(generated_token_ids)
            self._completions = self.model.tokenizer.decode(generated_token_ids)
        self._last_tokens = generated_token_ids

        masked_logits = []
        for idx in range(logits.shape[0]):
            masked = self._get_masked_logits(
                generated_token_ids[idx : idx + 1], logits[idx : idx + 1], idx
            )
            masked_logits.append(masked)

        return torch.cat(masked_logits, dim=0)


def cfg(
    model,
    cfg_string: str,
    max_tokens: Optional[int] = None,
    *,
    sampler: Optional["Sampler"] = None,
    allow_empty_tokens: bool = True,
):
    """Generate text sequences that match the input context-free grammar.

    Parameters
    ----------
    model
        The language model to use to compute the next-token logits.
    cfg_string
        The lark compliant cfg that generated expressions must match.
    max_tokens
        The maximum number of tokens to generate.
    sampler
        The function used to draw samples.  Defaults to
        `outlines.text.generate.sample.multinomial`.  See
        `outlines.text.generate.sample.Sampler` for the expected form of
        such functions.
    allow_empty_tokens
        Allow sampling of tokens corresponding to empty strings.

    .. note:
        Reuse instances of these guided generators whenever possible,
        because constructing them has more overhead than generating
        token sequences from them.  See the docstring for `CFG`.

    """
    return CFG(
        model,
        cfg_string,
        max_tokens,
        sampler=sampler,
        allow_empty_tokens=allow_empty_tokens,
    )
