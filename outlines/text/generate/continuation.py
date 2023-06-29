from typing import List, Optional

import torch

from outlines.text.generate.sequence import Sequence


class Continuation(Sequence):
    """Represents a completion generation model.

    `Completion` instances are unconstrained generation models that stop when an EOS token
    has been found or when the maximum number of tokens has been reached.

    >> import outlines.text as text
    >> sequence = text.sequence(model)("Say something")

    """

    def __init__(self, model, max_tokens: Optional[int]):
        super().__init__(model, max_tokens)
        self.eos_token_id = torch.tensor(
            [self.model.tokenizer.eos_token_id], device=self.device
        )

    def is_finished(self, token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Determine whether the sequences reached maximum length of end with
        and EOS token.

        In practice, `Sequence`'s `__call__` methods only passed the `token_ids`
        of the sequences that haven't been marked as finished already, which is
        why we only need to look for the EOS token in the last element rather
        than in the whole sequence.

        Parameters
        ----------
        token_ids
            The input sequences.

        """
        return token_ids[:, -1] == self.model.tokenizer.eos_token_id

    def postprocess_completions(self, completions: List[str]) -> List[str]:
        """Remove the EOS token from the completion."""
        return [
            completion.replace(self.model.tokenizer.eos_token, "")
            for completion in completions
        ]


def continuation(model, max_tokens: Optional[int] = None):
    return Continuation(model, max_tokens)
