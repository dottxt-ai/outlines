from typing import List, Optional, Tuple, Union

import torch

from outlines.text.generate.sequence import Sequence, vectorized_random_choice


def parallel_multi_step(
    model,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_prompt_tokens: int,
) -> torch.LongTensor:
    """Provides generation probabilities for an input sequence of tokens.
    Uses only one forward pass through the model.
    Output is a tensor of shape
    (batch_size, num_tokens - num_prompt_tokens + 1, vocab_size)
    representing generation probabilities for each non-prompt token
    and one additional next token.
    """
    # Perhaps we should raise an error here when the model does not provide all logits
    # e.g. gpt models
    logits = model.model(
        token_ids,
        attention_mask,
    ).logits

    # Will need to add something like the below for regex support
    # Problem is for regex continuation.create_proposal does in place modifications of
    # continuation.last_fsm_states and so we will need to be careful
    # logits = [continuation.create_proposal(token_ids[:, num_prompt_tokens:(num_prompt_tokens + i)], logits[..., i, :]) for i in range(len(logits.shape[-2]))]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    sampled_and_next_token_probs = probs[..., (num_prompt_tokens - 1) :, :]
    return sampled_and_next_token_probs


def serial_multi_step(
    continuation: Sequence,
    steps: int,
    rng: torch.Generator,
    num_prompt_tokens: int,
    token_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    is_finished: torch.BoolTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
    """Generates multiple tokens in serial.
    Also returns the full generation probabilities for each step.
    """
    dim_vocab = len(continuation.model.tokenizer.vocabulary)
    batch_shape = token_ids.shape[:-1]

    gen_probs = torch.ones(
        batch_shape
        + (
            steps,
            dim_vocab,
        ),
        device=continuation.device,
    )

    for i in range(steps):
        num_generated_tokens = token_ids.shape[-1] - num_prompt_tokens
        if torch.all(is_finished) or num_generated_tokens == continuation.max_tokens:
            break

        updated_token_ids, probs = continuation.step(
            rng,
            num_prompt_tokens,
            token_ids[~is_finished],
            attention_mask[~is_finished],
        )
        token_ids = continuation.update_token_ids(
            is_finished, token_ids, updated_token_ids
        )

        gen_probs[..., i, :] = probs

        attention_mask = continuation.expand_attention_mask(attention_mask)
        is_finished[~is_finished] = continuation.is_finished(
            updated_token_ids[:, num_prompt_tokens:]
        ).flatten()
    return token_ids, attention_mask, is_finished, gen_probs


class SpeculativeContinuation:
    """Generates a sequence using speculative decoding."""

    def __init__(self, draft_continuation, eval_model, speculation_length):
        self.draft_continuation = draft_continuation
        self.eval_model = eval_model
        self.speculation_length = speculation_length
        self.device = draft_continuation.device

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        samples: int = 1,
        rng: Optional[torch.Generator] = None,
    ) -> Union[str, List[str]]:
        """Generate a new sequence given a prompt.
        Using speculative decoding https://arxiv.org/abs/2211.17192
        Uses self.draft_continuation to generate a draft sequence and then
        self.eval_model for rejection sampling. The resulting sequence will be
        sampled according to the sampling distribution of self.eval_model.

        Parameters
        ----------
        prompt
            The input prompt.
        samples
            The number of samples to generate for each prompt.

        Returns
        -------
        The full sequence that contains the prompts and the generated string.

        """
        if samples > 1:
            raise NotImplementedError(
                "Sampling not implemented for speculative decoding"
            )

        token_ids, attention_mask = self.draft_continuation.model.tokenizer.encode(
            prompt
        )

        if len(token_ids) > 1:
            raise NotImplementedError(
                "Batching not implemented for speculative decoding"
            )

        token_ids = token_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        num_prompt_tokens = token_ids.shape[-1]

        batch_shape = token_ids.shape[:-1]
        is_finished = torch.zeros(batch_shape, dtype=torch.bool, device=self.device)

        while True:
            num_generated_tokens = token_ids.shape[-1] - num_prompt_tokens
            if (
                torch.all(is_finished)
                or num_generated_tokens == self.draft_continuation.max_tokens
            ):
                break

            steps = torch.min(
                torch.tensor(
                    [
                        self.speculation_length,
                        self.draft_continuation.max_tokens - num_generated_tokens,
                    ]
                )
            )

            (
                token_ids,
                attention_mask,
                is_finished_temp,
                gen_probs,
            ) = serial_multi_step(
                self.draft_continuation,
                steps,
                rng,
                num_prompt_tokens,
                token_ids,
                attention_mask,
                is_finished,
            )

            # Not vectorised from here on
            eval_probs = parallel_multi_step(
                self.eval_model,
                token_ids,
                attention_mask,
                token_ids.shape[-1] - steps,
            )

            token_gen_probs = gen_probs[
                ..., torch.arange(steps), token_ids[..., -steps:]
            ][0]

            token_eval_probs = eval_probs[
                ..., torch.arange(steps), token_ids[..., -steps:]
            ][0]

            accept_probs = token_eval_probs / token_gen_probs
            accept_samps = torch.rand(
                accept_probs.shape, generator=rng, device=self.device
            )
            reject_bools = accept_samps > accept_probs

            # Not vectorised
            keep_ind = torch.where(reject_bools[0])[0]

            if len(keep_ind) > 0:
                num_reject = steps - keep_ind
                token_ids = token_ids[:, :-num_reject]
                attention_mask = attention_mask[:, :-num_reject]
                pdash = eval_probs[..., keep_ind, :] - gen_probs[..., keep_ind, :]
                pdash = torch.where(pdash < 0, torch.zeros_like(pdash), pdash)
                pdash = pdash / torch.sum(pdash, dim=-1, keepdim=True)

                sampled_tok = vectorized_random_choice(rng, pdash)

                token_ids = torch.concatenate([token_ids, sampled_tok], axis=-1)

            elif (
                is_finished_temp.all()
                or num_generated_tokens + steps == self.draft_continuation.max_tokens
            ):
                break

            else:
                eval_next_token_probs = eval_probs[..., -1, :]
                sampled_next_token = vectorized_random_choice(
                    rng, eval_next_token_probs
                )
                token_ids = torch.concatenate([token_ids, sampled_next_token], axis=-1)

            attention_mask = self.draft_continuation.expand_attention_mask(
                attention_mask
            )
            is_finished[~is_finished] = self.draft_continuation.is_finished(
                token_ids[:, num_prompt_tokens:]
            ).flatten()

        result = self.draft_continuation.model.tokenizer.decode(
            token_ids[..., num_prompt_tokens:]
        )
        result = self.draft_continuation.postprocess_completions(result)

        if len(result) == 1:
            return result[0]

        return result


def speculative_decode(
    draft_continuation: Sequence, eval_model, speculation_length: int
) -> SpeculativeContinuation:
    """Generates a sequence using speculative decoding.
    For more details see https://arxiv.org/abs/2211.17192.

    Parameters
    ----------
    draft_continuation
        The outlines Sequence object from which to generate draft tokens
        (i.e. with the faster, draft model to be executed in serial).
    eval_model
        The slower, better model to be executed in parallel.
        The final output will be distributed according to eval_model.
    speculation_length
        Determines how many tokens to generate in serial before applying
        rejection sampling with eval_model.

    """
    return SpeculativeContinuation(draft_continuation, eval_model, speculation_length)
