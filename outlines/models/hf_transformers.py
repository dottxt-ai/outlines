"""Integration with HuggingFace's `transformers` library."""
from typing import Callable, Optional

from outlines.caching import cache


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

    def call(prompt: str, *, samples: int = 1, type: Optional[str] = None) -> str:
        return call_model_generate_method(
            model_name, prompt, max_tokens, temperature, samples, type
        )

    return call


@cache
def call_model_generate_method(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    samples: int,
    type: str,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt_tokens = tokenizer(prompt, return_tensors="pt")

    logit_processor, stopping_criterion = None, None
    if type is not None:
        if samples > 1:
            raise NotImplementedError(
                "It is currently not possible to control the generation of several samples with the `transformers` integration"
            )
        logit_processor, stopping_criterion = create_type_mask(
            type, tokenizer, prompt_tokens["input_ids"]
        )

    if torch.cuda.is_available():
        model = model.to("cuda")
        prompt_tokens = prompt_tokens.to("cuda")

    returned_tokens = model.generate(
        **prompt_tokens,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=samples,
        logits_processor=[logit_processor],
        stopping_criteria=[stopping_criterion],
    )
    new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] + 1 :]
    new_tokens = new_tokens.squeeze()

    if samples == 1:
        results = tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        results = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return results


def create_int_mask(tokenizer, prompt_tokens):
    """TODO: Make sure that we catch all cases."""
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]

    mask = torch.zeros(len(tokenizer), dtype=torch.bool)

    for token, token_id in tokenizer.get_vocab().items():
        token = tokenizer.decode(token_id)
        are_all_digits = all([c.isdigit() for c in token])
        if are_all_digits:
            mask[token_id] = True

    mask[tokenizer.eos_token_id] = False

    def processor(input_ids, scores):
        if input_ids.shape[1] > num_prompt_tokens + 1:
            mask[tokenizer.eos_token_id] = True
        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")
        return scores

    def stopping_criterion(input_ids, _):
        decoded_input = tokenizer.decode(
            input_ids[0, num_prompt_tokens:], skip_special_tokens=True
        )
        is_starting_new_sequence = all([c.isdigit() for c in decoded_input]) and (
            decoded_input[-1] == " "
            or decoded_input[-1] == "\n"
            or decoded_input[-1] == "\r"
        )
        if len(decoded_input) > 1 and is_starting_new_sequence:
            return True

        return False

    return processor, stopping_criterion


def create_float_mask(tokenizer, prompt_tokens, decimals=3):
    """TODO: Make sure that we catch all cases."""
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]

    mask = torch.zeros(len(tokenizer), dtype=torch.bool)

    for token, token_id in tokenizer.get_vocab().items():
        token = tokenizer.decode(token_id)
        is_valid_float_or_int = (
            all([c.isdigit() or c == "." for c in token]) and token.count(".") <= 1
        )
        if is_valid_float_or_int:
            mask[token_id] = True

    mask[tokenizer.eos_token_id] = False

    def processor(input_ids, scores):
        if input_ids.shape[1] > num_prompt_tokens + 1:
            mask[tokenizer.eos_token_id] = True
        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")
        return scores

    def stopping_criterion(input_ids, _):
        decoded_input = tokenizer.decode(
            input_ids[0, num_prompt_tokens:], skip_special_tokens=True
        )
        if decoded_input.count(".") > 1:
            return True

        if (
            decoded_input.count(".") == 1
            and len(decoded_input.strip().split(".")[1]) > decimals
        ):
            return True

        if len(decoded_input) > 1:
            is_starting_new_sequence = all(
                [c.isdigit() for c in decoded_input[:-1]]
            ) and (
                decoded_input[-1] == " "
                or decoded_input[-1] == "\n"
                or decoded_input[-1] == "\r"
            )
            if is_starting_new_sequence:
                return True

        return False

    return processor, stopping_criterion


type_to_mask = {
    "float": create_float_mask,
    "int": create_int_mask,
}


def create_type_mask(type, tokenizer, prompt_tokens):
    if type not in ["int", "float"]:
        raise NotImplementedError(f"Cannot restrict the generation to type {type}")

    return type_to_mask[type](tokenizer, prompt_tokens)
