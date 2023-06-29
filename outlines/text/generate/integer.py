import math
from typing import List, Optional, Tuple

import interegular
import torch

from outlines.text.generate.continuation import Continuation
from outlines.text.parsing import find_partial_matches, map_partial_states_to_vocab


class Integer(Continuation):
    """Represents a integer generation model.

    `Integer` instances are constrained generation models that only
    generate integer values. Leading zeros are fobidden. EOS tokens
    are only allowed after at least one digit has been generated.

    >>> import outlines.text as text
    >>> sequence = text.generate.integer(model)("Return an integer between 0 and 10")

    """

    def __init__(self, model, max_tokens: Optional[int]):
        super().__init__(model, max_tokens)

        vocabulary = model.tokenizer.vocabulary
        sorted_vocabulary = [
            k for k, v in sorted(vocabulary.items(), key=lambda kv: kv[1])
        ]

        int_regex_string = r"(0|[1-9][0-9]+)"
        int_regex_pattern = interegular.parse_pattern(int_regex_string)
        self.int_regex_fsm = int_regex_pattern.simplify().to_fsm()

        def partial_match_filter(string, end_idx, state_seq):
            if end_idx is not None and end_idx < len(string) - 1:
                return False
            return True

        pstate_to_vocab = map_partial_states_to_vocab(
            list(sorted_vocabulary),
            {"INT": self.int_regex_fsm},
            True,
            partial_match_filter,
        )
        self.pstate_to_vocab = {k: list(v) for k, v in pstate_to_vocab.items()}

    def create_proposal(
        self, generated_token_ids: torch.LongTensor, logits: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Modify the next-token logits so that only integers can be generated.

        Parameters
        ----------
        generated_token_ids
            The token ids generated so far.
        logits
            The next-token logits.

        """
        if generated_token_ids.shape[-1] > 0:
            # TODO Make this work for `generated_token_ids` of arbitrary shape
            sampled_sequences = self.model.tokenizer.decode(generated_token_ids)
            if isinstance(sampled_sequences, str):
                sampled_sequences = [sampled_sequences]
            partial_matches = [
                find_partial_matches(self.int_regex_fsm, sequence)
                for sequence in sampled_sequences
            ]
            pmatches = [
                max(partial_match, key=lambda x: x[0] if x[0] is not None else -1)
                for partial_match in partial_matches
            ]
            self.pstates: List[Tuple[str, int]] = [
                (self.pstates[0][0], pmatch[1][-1]) for pmatch in pmatches
            ]
        else:
            self.pstates = [
                ("INT", self.int_regex_fsm.initial)
                for _ in range(generated_token_ids.shape[0])
            ]

        masks = []
        for pstate in self.pstates:
            next_support = self.pstate_to_vocab[pstate]
            mask = torch.full((len(self.model.tokenizer.vocabulary),), -math.inf)
            mask[next_support] = 0
            masks.append(mask.unsqueeze(0))

        mask = torch.concatenate(masks, dim=0)

        return logits + mask


def integer(model, max_tokens: Optional[int] = None):
    return Integer(model, max_tokens)
