from typing import Tuple, List, Optional, Union
import inspect
import torch

from outlines.text.generate.continuation import Continuation
from outlines.text.generate.sequence import Sequence


def parallel_multi_step(
    continuation: Continuation,
    token_ids: torch.Tensor,
    num_prompt_tokens: int,
    batch_size: int = None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    token_ids_sqz = token_ids.squeeze()
    gen_toks = token_ids_sqz[num_prompt_tokens:]
    gen_toks = torch.concatenate([gen_toks, torch.zeros(1, dtype=torch.int64, device=continuation.device)])

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
        pseudo_prompt = token_ids_sqz[: (num_prompt_tokens + i)]
        in_tokens[i, (-len(pseudo_prompt)) :] = pseudo_prompt
        attention_masks[i, (-len(pseudo_prompt)) :] = 1

    sampled_probs = torch.zeros(
        num_generated_tokens + 1, dtype=torch.float32, device=continuation.device
    )
    
    sampled_token_ids = torch.zeros(
        num_generated_tokens + 1, dtype=torch.float32, device=continuation.device
    )

    # Run in batches
    for j in range(-((num_generated_tokens + 1) // -batch_size)):
        start = j * batch_size
        end = min((j + 1) * batch_size, num_generated_tokens + 1)

        batch_sampled_token_ids, eval_probs = continuation.step(
            rng,
            num_prompt_tokens,
            in_tokens[start:end],
            attention_masks[start:end],
        )
        sampled_toks_temp = batch_sampled_token_ids[..., -1]
        sampled_token_ids[start:end] = sampled_toks_temp
        
        if end == num_generated_tokens + 1:
            gen_toks[-1] = sampled_toks_temp[-1]
        
        sampled_probs_temp = eval_probs[torch.arange(end - start), gen_toks[start:end]]
        sampled_probs[start:end] = sampled_probs_temp

    return sampled_token_ids, sampled_probs


def serial_multi_step(continuation, steps, rng, num_prompt_tokens, token_ids, attention_mask, is_finished, gen_probs):
    
    for _ in range(steps):
        num_generated_tokens = token_ids.shape[-1] - num_prompt_tokens
        if torch.all(is_finished) or num_generated_tokens == continuation.max_tokens:
            break

        updated_token_ids, probs = continuation.step(
            rng,
            num_prompt_tokens,
            token_ids[~is_finished],
            attention_mask[~is_finished],
        )
        token_ids = continuation.update_token_ids(is_finished, token_ids, updated_token_ids)
        
        gen_probs = torch.concatenate([gen_probs, probs[0][token_ids[0, -1:]]])

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
                raise NotImplementedError("Sampling not implemented for speculative decoding")
            
            token_ids, attention_mask = self.model.tokenizer.encode(prompt)

            token_ids = token_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            if rng is None:
                rng = torch.Generator(device=self.device)
                rng.seed()

            num_prompt_tokens = token_ids.shape[-1]

            batch_shape = token_ids.shape[:-1]
            is_finished = torch.zeros(batch_shape, dtype=torch.bool, device=self.device)
            gen_probs = torch.ones(num_prompt_tokens, device=self.device)

            while True:
                num_generated_tokens = token_ids.shape[-1] - num_prompt_tokens
                if torch.all(is_finished) or num_generated_tokens == self.max_tokens:
                    break
                
                steps = torch.min(torch.tensor([self.speculation_length,
                                                self.max_tokens - num_generated_tokens]))
                token_ids, attention_mask, is_finished, gen_probs = multi_step(self,
                    steps, rng, num_prompt_tokens, token_ids, attention_mask, is_finished, gen_probs
                )
                
                eval_token_ids, eval_probs = token_ids_to_probs(
                    self.eval_model_continuation,
                    token_ids,
                    token_ids.shape[-1] - steps,
                    steps,
                    rng,
                )
                
                accept_probs = eval_probs / gen_probs
                accept_samps = torch.rand(accept_probs.shape, generator=rng, device=self.device)
                reject_bools = accept_samps > accept_probs
                keep_ind = torch.where(reject_bools)[0]
                
                if len(keep_ind) == 0 and not torch.all(is_finished):

                    
                else:
                    n_trim = steps - len(keep_ind)
                    token_ids = token_ids[:-n_trim]
                    attention_mask = attention_mask[:-n_trim]
                    gen_probs = gen_probs[:-n_trim]
                    
                is_finished[~is_finished] = self.is_finished(
                    token_ids[~is_finished, num_prompt_tokens:]
                ).flatten()
                
                
                

            result = self.model.tokenizer.decode(token_ids[..., num_prompt_tokens:])
            result = self.postprocess_completions(result)

            if len(result) == 1:
                return result[0]

            return result

    return SpeculativeContinuation()


if __name__ == "__main__":
    import outlines.models as models
    from outlines import text
    import torch

    # draft_model = models.transformers("meta-llama/Llama-2-7b-hf", device="cuda")
    # eval_model = models.transformers("meta-llama/Llama-2-13b-hf", device="cuda")
    
    draft_model = models.transformers("gpt2")
    draft_model.model.eval()
    eval_model = draft_model
    

    # prompt = "The prime factors of 14 are"
    prompt = "The best Harry Potter book is"
    mts = 20
    stop = ["\n", "."]

    continuation = text.generate.continuation(draft_model, max_tokens=mts, stop=stop)

    
    self = continuation
    self.eval_model_continuation = continuation
    self.speculation_length = 5
        
    
    token_ids, attention_mask = self.model.tokenizer.encode(prompt)

    token_ids = token_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    num_prompt_tokens = token_ids.shape[-1]

    batch_shape = token_ids.shape[:-1]
    is_finished = torch.zeros(batch_shape, dtype=torch.bool, device=self.device)
    gen_probs = torch.ones(num_prompt_tokens, device=self.device)
    
    s_toks, _, _, s_probs = serial_multi_step(continuation, 5, torch.manual_seed(0), num_prompt_tokens, token_ids, attention_mask, is_finished, gen_probs)
    
    p_toks, p_probs = parallel_multi_step(continuation, s_toks, num_prompt_tokens, rng=torch.manual_seed(0))
    
    
    
    
    rng = torch.manual_seed(0)
    continuation(prompt, rng=rng)
    
    rng = torch.manual_seed(0)
    continuation(prompt, rng=rng)
    
    supervised_continuation = speculative_decode(continuation, eval_model)

    output_sup = supervised_continuation(prompt)
