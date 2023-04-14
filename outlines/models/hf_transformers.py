"""Integration with HuggingFace's `transformers` library."""
from typing import Callable, Optional

import outlines.cache as cache

memory = cache.get()


def HuggingFaceCompletion(
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Callable:
    """Create a function that will call the `generate` method of a `transformers` model.

    You should have the `torch` and `transformers` packages installed. First
    execution may take a while since the pre-trained weights will be downloaded.
    Available models are listed on `HuggingFace's model page <https://huggingface.co/models>`_.

    Note
    ----

    To my knowledge `tranformers` does not simply allow to stop the generation
    after a given sequence has been generated. We will need to implement this
    manually for this integration to have the same features as `OpenAICompletion`.

    Parameters
    ----------
    model_name: str
        The name of the model as listed on HuggingFace's models page.
    max_tokens
        The maximum number of tokens to generate.
    temperature
        Value used to module the next token probabilities.

    Returns
    -------
    A function that will generate tokens from the model when passed a prompt.

    """
    if max_tokens is None:
        max_tokens = 216

    if temperature is None:
        temperature = 1.0

    def call(prompt: str) -> str:
        return call_model_generate_method(model_name, prompt, max_tokens, temperature)

    return call


@memory.cache
def call_model_generate_method(
    model_name: str, prompt: str, max_tokens: int, temperature: float
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt_tokens = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        model = model.to("cuda")
        prompt_tokens = prompt_tokens.to("cuda")

    returned_tokens = model.generate(
        **prompt_tokens,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] + 1 :]
    new_tokens = new_tokens.squeeze()

    return tokenizer.decode(new_tokens, skip_special_tokens=True)
