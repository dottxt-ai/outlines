"""Integration with HuggingFace's `transformers` library."""
from typing import Callable, List, Optional, Union

import outlines.cache as cache

memory = cache.get()


def HuggingFaceCompletion(
    model_name: str,
    max_tokens: Optional[int] = 216,
    temperature: Optional[float] = 1.0,
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

    def call(prompt: Union[str, List[str]], num_samples: int = 1) -> str:
        return call_model_generate_method(
            model_name, prompt, num_samples, max_tokens, temperature
        )

    return call


@memory.cache()
def call_model_generate_method(
    model_name: str,
    prompt: Union[str, List[str]],
    num_samples: int,
    max_tokens: int,
    temperature: float,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if isinstance(prompt, str):
        prompt = [prompt]

    tokenizer.pad_token = tokenizer.eos_token
    prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True)

    if torch.cuda.is_available():
        model = model.to("cuda")
        prompt_tokens = prompt_tokens.to("cuda")

    returned_tokens = model.generate(
        **prompt_tokens,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_samples,
    )

    new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] + 1 :]
    new_tokens = new_tokens.squeeze()
    new_sequences = decode(tokenizer, num_samples, new_tokens)

    return new_sequences


def decode(tokenizer, num_samples, new_tokens):
    """Decode the output tokens.

    The model's `generate` method will always return an array with at most
    two dimensions, even with batched inputs and several samples. We thus
    need to reshape the output list to return nested lists.

    """
    if new_tokens.ndim == 1:
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # Reshape the list if we have several samples
        if num_samples > 1:
            decoded = [
                decoded[i : i + num_samples]
                for i in range(0, len(decoded), num_samples)
            ]
            if len(decoded) == 1:
                decoded = decoded[0]
        return decoded
