from typing import List, Optional, Union

import torch

from outlines.text.generate.sequence import Sequence


class Continuation(Sequence):
    """Represents a completion generation model.

    `Continuation` instances are unconstrained generation models that stop when
    an EOS token has been found or when the maximum number of tokens has been
    reached.

    >>> import outlines.text as text
    >>> sequence = text.generate.continuation(model)("Say something")

    """

    def __init__(
        self, model, max_tokens: Optional[int] = None, stop: Union[str, List[str]] = []
    ):
        super().__init__(model, max_tokens)
        self.eos_token_id = torch.tensor(
            [self.model.tokenizer.eos_token_id], device=self.device
        )

        if isinstance(stop, str):
            stop = [stop]

        self.stop_sequences = stop

    def is_finished(self, token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Determine whether the sequences reached maximum length of end with
        and EOS token.

        We only need to look for the EOS token in the last element rather than
        in the whole sequence. Indeed, (1) EOS is a single token (2)
        `Sequence`'s `__call__` methods only passed the `token_ids` of the
        sequences that haven't been marked as finished already.

        Parameters
        ----------
        token_ids
            The input sequences.

        """

        sequences = self.model.tokenizer.decode(token_ids)
        contains_stop_sequence = []
        for sequence in sequences:
            found = False
            for stop_str in self.stop_sequences:
                if stop_str in sequence:
                    found = True

            contains_stop_sequence.append(found)

        contains_stop_sequence = torch.tensor(
            contains_stop_sequence, dtype=torch.bool, device=self.model.device
        )
        contains_eos = token_ids[:, -1] == self.model.tokenizer.eos_token_id

        return torch.logical_or(contains_eos, contains_stop_sequence)

    def postprocess_completions(self, completions: List[str]) -> List[str]:
        """Remove the EOS token from the completion.

        Sequences in `stop` take precedence over EOS. For instance, if
        `stop=["\n"]` and the generated sequence is 'One\nTwo<EOS>`
        `Continuation.postprocess_completions` will return `One`.

        """
        completions_without_eos = [
            completion.replace(self.model.tokenizer.eos_token, "")
            for completion in completions
        ]

        completions_without_stop = []
        for completion in completions_without_eos:
            for stop_str in self.stop_sequences:
                idx = completion.rfind(stop_str)  # ignore the prompt
                if idx > 0:
                    completion = completion[:idx]

            completions_without_stop.append(completion)

        return completions_without_stop


def continuation(
    model, max_tokens: Optional[int] = None, *, stop: Union[str, List[str]] = []
):
    """Generate text sequences.

    Parameters
    ----------
    model
        The model to use to computes the next-token logits.
    max_tokens
        The maximum number of tokens to generate.
    stop
        A string or list of strings which, when generated, stops
        the generation for this sequence.

    """
    return Continuation(model, max_tokens, stop)
