"""Integration with HuggingFace's `transformers` library."""
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from outlines.caching import cache

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


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

    def call(
        prompt: str,
        *,
        samples: int = 1,
        stop_at: Optional[List[str]] = None,
        is_in: Optional[List[str]] = None,
        type: Optional[str] = None,
    ) -> str:
        return call_model_generate_method(
            model_name, prompt, max_tokens, temperature, samples, stop_at, is_in, type
        )

    return call


@cache
def call_model_generate_method(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    samples: int,
    stop_at: List[str],
    is_in: List[str],
    type: str,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt_tokens = tokenizer(prompt, return_tensors="pt")

    logit_processors: Optional[List[Callable]] = None
    stopping_criteria: Optional[List[Callable]] = None
    postprocessing: Callable = lambda x: x
    if type is not None:
        if samples > 1:
            raise NotImplementedError(
                "It is currently not possible to control the generation of several samples with the `transformers` integration"
            )
        if is_in is not None:
            raise ValueError(
                "You cannot both restrict to a set of choices with `is_in` and to a type with `type`"
            )
        logit_processor, stopping_criterion, postprocessing = create_type_mask(
            type, tokenizer, prompt_tokens["input_ids"]
        )
        logit_processors = [logit_processor]
        stopping_criteria = [stopping_criterion]
    elif is_in is not None:
        if stop_at is not None:
            raise ValueError(
                "You cannot both restrict to a set of choices with `is_in` and set a stopping criterion"
            )
        logit_processor, stopping_criterion, postprocessing = create_choice_mask(
            is_in, tokenizer, prompt_tokens["input_ids"]
        )
        logit_processors = [logit_processor]
        stopping_criteria = [stopping_criterion]

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
        logits_processor=logit_processors,
        stopping_criteria=stopping_criteria,
    )
    new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] :]
    new_tokens = new_tokens.squeeze()

    if samples == 1:
        results = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results = postprocessing(results)
    else:
        results = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return results


def create_choice_mask(
    choices: List[str],
    tokenizer: "PreTrainedTokenizerBase",
    prompt_tokens: "torch.Tensor",
) -> Tuple[Callable, Callable, Callable]:
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]
    tokenized_choices = [tokenizer.encode(word) for word in choices]

    def processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        output = input_ids[0, num_prompt_tokens:]
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)

        mask = torch.zeros(len(tokenizer), dtype=torch.bool)
        for choice, tokens in zip(choices, tokenized_choices):
            if not choice.startswith(decoded_output):
                continue
            else:
                mask[tokens[len(output)]] = True

        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")

        return scores

    def stopping_criterion(input_ids: torch.Tensor, _) -> bool:
        """
        TODO: We can stop the generation once we have excluded all possibilities but one, and the
        full sequence can be recovered during post-processing.
        """
        decoded_input = tokenizer.decode(
            input_ids[0, num_prompt_tokens:], skip_special_tokens=True
        )
        if decoded_input in choices:
            return True

        return False

    def postprocessing(output: str) -> str:
        return output

    return processor, stopping_criterion, postprocessing


def create_int_mask(
    tokenizer: "PreTrainedTokenizerBase", prompt_tokens: "torch.Tensor"
) -> Tuple[Callable, Callable, Callable]:
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]

    mask = torch.zeros(len(tokenizer), dtype=torch.bool)

    for _, token_id in tokenizer.get_vocab().items():
        token = tokenizer.decode(token_id)
        are_all_digits = all([c.isdigit() for c in token])
        if are_all_digits:
            mask[token_id] = True

    mask[tokenizer.eos_token_id] = False

    def processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > num_prompt_tokens + 1:
            mask[tokenizer.eos_token_id] = True
        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")
        return scores

    def stopping_criterion(input_ids: torch.Tensor, _) -> bool:
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

    def postprocessing(output: str) -> str:
        return output

    return processor, stopping_criterion, postprocessing


def create_float_mask(
    tokenizer: "PreTrainedTokenizerBase",
    prompt_tokens: "torch.Tensor",
    decimals: int = 3,
) -> Tuple[Callable, Callable, Callable]:
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]

    mask = torch.zeros(len(tokenizer), dtype=torch.bool)

    for _, token_id in tokenizer.get_vocab().items():
        token = tokenizer.decode(token_id)
        is_valid_float_or_int = (
            all([c.isdigit() or c == "." for c in token]) and token.count(".") <= 1
        )
        if is_valid_float_or_int:
            mask[token_id] = True

    mask[tokenizer.eos_token_id] = False

    def processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > num_prompt_tokens + 1:
            mask[tokenizer.eos_token_id] = True
        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")
        return scores

    def stopping_criterion(input_ids: torch.Tensor, _) -> bool:
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

    def postprocessing(output: str) -> str:
        return output.rstrip(".")

    return processor, stopping_criterion, postprocessing


type_to_mask: Dict[str, Callable] = {
    "float": create_float_mask,
    "int": create_int_mask,
}


def create_type_mask(
    type: str, tokenizer: "PreTrainedTokenizerBase", prompt_tokens: "torch.Tensor"
) -> Tuple[Callable, Callable, Callable]:
    if type not in ["int", "float"]:
        raise NotImplementedError(f"Cannot restrict the generation to type {type}")

    return type_to_mask[type](tokenizer, prompt_tokens)
