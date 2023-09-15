from typing import Tuple, List, Optional, Union
import inspect
import torch

from outlines.text.generate.continuation import Continuation
from outlines.text.generate.sequence import Sequence, vectorized_random_choice


def parallel_multi_step(
    continuation: Continuation,
    token_ids: torch.Tensor,
    num_prompt_tokens: int,
    batch_size: int = None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    
    if token_ids.ndim != 1:
        raise NotImplementedError("Vectorised parallel_multi_step not implemented")
    
    dim_vocab = len(continuation.model.tokenizer.vocabulary)

    num_tokens = token_ids.shape[-1]
    num_generated_tokens = num_tokens - num_prompt_tokens

    if batch_size is None:
        batch_size = num_generated_tokens + 1

    if rng is None:
        rng = torch.Generator(device=continuation.device)
        rng.seed()

    in_tokens = torch.zeros(
        (num_generated_tokens + 1, num_tokens),
        dtype=torch.int64,
        device=continuation.device,
    )
    attention_masks = torch.zeros_like(
        in_tokens, dtype=torch.int64, device=continuation.device
    )
    for i in range(num_generated_tokens + 1):
        pseudo_prompt = token_ids[: (num_prompt_tokens + i)]
        in_tokens[i, (-len(pseudo_prompt)) :] = pseudo_prompt
        attention_masks[i, (-len(pseudo_prompt)) :] = 1

    sampled_token_ids = torch.zeros(
        num_generated_tokens + 1, dtype=torch.int64, device=continuation.device
    )

    eval_probs = torch.zeros(
        (num_generated_tokens + 1, dim_vocab),
        dtype=torch.float32,
        device=continuation.device,
    )

    # Run in batches
    for j in range(-((num_generated_tokens + 1) // -batch_size)):
        start = j * batch_size
        end = min((j + 1) * batch_size, num_generated_tokens + 1)

        batch_sampled_token_id_seqs, batch_eval_probs = continuation.step(
            rng,
            num_prompt_tokens,
            in_tokens[start:end],
            attention_masks[start:end],
        )

        sampled_toks_temp = batch_sampled_token_id_seqs[..., -1]
        sampled_token_ids[start:end] = sampled_toks_temp

        eval_probs[start:end] = batch_eval_probs

    return sampled_token_ids, eval_probs


def serial_multi_step(
    continuation: Sequence,
    steps: int,
    rng: torch.Generator,
    num_prompt_tokens: int,
    token_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    is_finished: torch.BoolTensor,
):
    dim_vocab = len(continuation.model.tokenizer.vocabulary)
    batch_shape = token_ids.shape[:-1]
    
    gen_probs = torch.ones(batch_shape + (steps, dim_vocab,), device=continuation.device)

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

        gen_probs[...,  i, :] = probs

        attention_mask = continuation.expand_attention_mask(attention_mask)
        is_finished[~is_finished] = continuation.is_finished(
            updated_token_ids[:, num_prompt_tokens:]
        ).flatten()
    return token_ids, attention_mask, is_finished, gen_probs


def speculative_decode(
    continuation: Sequence, eval_model, speculation_length: int, **kwargs
) -> Sequence:
    assert (
        continuation.model.tokenizer.tokenizer.__class__
        == eval_model.tokenizer.tokenizer.__class__
    ), "Model tokenizers are different"

    # Extract arguments required to create a new instance of continuation
    # So that we can create a new instance of continuation just using eval_model instead
    # This is hacky and I don't like it, perhaps we can find a better way?
    # The goal is to support speculative decoding for any continuation (e.g. regex)
    draft_init_args = inspect.signature(continuation.__init__).parameters
    draft_kwargs = {
        arg: getattr(continuation, arg) for arg in draft_init_args if arg != "stop"
    }
    draft_kwargs["stop"] = continuation.stop_sequences

    eval_kwargs = draft_kwargs | kwargs
    eval_kwargs["max_tokens"] = 1
    eval_kwargs["model"] = eval_model

    class SpeculativeContinuation(type(continuation)):
        def __init__(self):
            super().__init__(**draft_kwargs)
            self.eval_model_continuation = type(continuation)(**eval_kwargs)
            self.speculation_length = speculation_length

        @torch.inference_mode()
        def __call__(
            self,
            prompt: Union[str, List[str]],
            samples: int = 1,
            rng: Optional[torch.Generator] = None,
        ) -> Union[str, List[str]]:
            """Generate a new sequence given a prompt.
            Using speculative decoding https://arxiv.org/abs/2211.17192

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

            token_ids, attention_mask = self.model.tokenizer.encode(prompt)
            
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
                if torch.all(is_finished) or num_generated_tokens == self.max_tokens:
                    break

                steps = torch.min(
                    torch.tensor(
                        [
                            self.speculation_length,
                            self.max_tokens - num_generated_tokens,
                        ]
                    )
                )
                
                
                (
                    token_ids,
                    attention_mask,
                    is_finished_temp,
                    gen_probs,
                ) = serial_multi_step(
                    self,
                    steps,
                    rng,
                    num_prompt_tokens,
                    token_ids,
                    attention_mask,
                    is_finished
                )

                # Not vectorised from here on
                eval_token_ids, eval_probs = parallel_multi_step(
                    self.eval_model_continuation,
                    token_ids[0],
                    token_ids.shape[-1] - steps,
                    rng=rng,
                )
                eval_token_ids = eval_token_ids[None]
                eval_probs = eval_probs[None]
                

                token_gen_probs = gen_probs[
                    ..., torch.arange(steps), token_ids[..., -steps:]
                ][0]
                token_eval_probs = gen_probs[
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

                elif is_finished_temp.all():
                    break

                else:
                    token_ids = torch.concatenate(
                        [token_ids, eval_token_ids[:, -1:]], axis=-1
                    )

                attention_mask = self.expand_attention_mask(attention_mask)
                is_finished[~is_finished] = self.is_finished(
                    token_ids[:, num_prompt_tokens:]
                ).flatten()

            result = self.model.tokenizer.decode(token_ids[..., num_prompt_tokens:])
            result = self.postprocess_completions(result)

            if len(result) == 1:
                return result[0]

            return result

    return SpeculativeContinuation()

