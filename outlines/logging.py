import logging
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


LOG_LEVEL_OFF = logging.CRITICAL + 100
logits_logger = logging.getLogger("logits_logger")
logits_logger.setLevel(LOG_LEVEL_OFF)
logits_logger.addHandler(logging.StreamHandler(sys.stderr))


def enable_logits_logging():
    logits_logger.setLevel(logging.DEBUG)


def log_logits(
    tokenizer: "Tokenizer",
    logits: torch.Tensor,
    next_token_ids: torch.Tensor,
    top_n: int = 8,
):
    """ """

    def only(l):
        assert len(list(l)) == 1
        return l[0]

    # this function is expensive, skip it if logging isn't enabled
    if logits_logger.getEffectiveLevel() >= LOG_LEVEL_OFF:
        return

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        next_token_ids = next_token_ids.unsqueeze(0)

    batch_size = logits.size(0)

    for b in range(batch_size):
        current_logits = logits[b]
        next_token_id = next_token_ids[b]

        # all token probs for the current batch
        probs = torch.nn.functional.softmax(current_logits, dim=-1)

        # top candidate tokens probs
        top_probs, top_indices = torch.topk(probs, top_n)
        top_tokens_detail_str = ", ".join(
            [
                f"{repr(only(tokenizer.decode([idx])))}: {prob.item():.3f}"
                for prob, idx in zip(top_probs, top_indices)
            ]
        )

        # tokens from current_next_token_ids
        next_token = only(tokenizer.decode(next_token_id))

        # Log the information for the current batch
        logits_logger.info(f"Selected: {repr(next_token)} for batch_item={b}")
        logits_logger.info(
            f"\tEOS Prob: {probs[tokenizer.eos_token_id].item()}"
            + f"(logit = {current_logits[tokenizer.eos_token_id].item()})"
        )
        logits_logger.info(f"\tTop {top_n} Tokens: {top_tokens_detail_str}")
