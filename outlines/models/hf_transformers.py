"""Integration with Hugging Face's `transformers` library."""
import functools
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import outlines
from outlines.text.masks import create_float_mask, create_int_mask

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
    Available models are listed on `Hugging Face's model page <https://huggingface.co/models>`_.

    Note
    ----

    To my knowledge `transformers` does not simply allow to stop the generation
    after a given sequence has been generated. We will need to implement this
    manually for this integration to have the same features as `OpenAICompletion`.

    Parameters
    ----------
    model_name: str
        The name of the model as listed on Hugging Face's models page.
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
        prompt: Union[str, List[str]],
        *,
        samples: int = 1,
        stop_at: List[Optional[str]] = [],
        is_in: List[Optional[str]] = [],
        type: Optional[str] = None,
    ) -> str:
        if isinstance(prompt, str):
            prompt = [prompt]

        return call_model_generate_method(
            model_name,
            prompt,
            max_tokens,
            temperature,
            samples,
            stop_at,
            is_in,
            type,
        )

    return call


@functools.partial(outlines.vectorize, signature="(),(m),(),(),(),(i),(j),()->(m,s)")
def call_model_generate_method(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    samples: int,
    stop_at: List[Optional[str]],
    is_in: np.ndarray,
    type: str,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # `generate` does not accept NumPy arrays
    prompt = list(prompt)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="left")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True)

    logit_processors: Optional[List[Callable]] = None
    stopping_criteria: Optional[List[Callable]] = None
    postprocessing: Callable = lambda x: x
    if type is not None:
        if samples > 1:
            raise NotImplementedError(
                "It is currently not possible to control the generation of several samples with the `transformers` integration"
            )
        if is_in.size > 0:
            raise ValueError(
                "You cannot both restrict to a set of choices with `is_in` and to a type with `type`"
            )
        logit_processor, stopping_criterion, postprocessing = create_type_constraint(
            type, tokenizer, prompt_tokens["input_ids"]
        )
        logit_processors = [logit_processor]
        stopping_criteria = [stopping_criterion]
    elif is_in.size > 0:
        if samples > 1:
            raise NotImplementedError(
                "It is currently not possible to control the generation of several samples with the `transformers` integration"
            )
        if stop_at.size > 0:
            raise ValueError(
                "You cannot both restrict to a set of choices with `is_in` and set a stopping criterion"
            )
        logit_processor, stopping_criterion, postprocessing = create_choice_constraint(
            is_in, tokenizer, prompt_tokens["input_ids"]
        )
        logit_processors = [logit_processor]
        stopping_criteria = [stopping_criterion]
    elif stop_at.size > 0:
        if samples > 1:
            raise NotImplementedError(
                "It is currently not possible to control the generation of several samples with the `transformers` integration"
            )
        logit_processor, stopping_criterion, postprocessing = create_stop_constraint(
            stop_at, tokenizer, prompt_tokens["input_ids"]
        )
        logit_processors = [logit_processor]
        stopping_criteria = [stopping_criterion]

    if torch.cuda.is_available():
        model = model.to("cuda")
        prompt_tokens = prompt_tokens.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
        prompt_tokens = prompt_tokens.to("mps")

    returned_tokens = model.generate(
        **prompt_tokens,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=int(samples),
        logits_processor=logit_processors,
        stopping_criteria=stopping_criteria,
    )
    new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] :]
    if len(prompt) == 1:
        new_tokens = new_tokens.squeeze()

    if new_tokens.ndim < 2:
        results = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results = np.array([postprocessing(results)])
    else:
        results = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        results = [postprocessing(result) for result in results]
        results = np.array(results)

    if len(prompt) == 1:
        results = np.expand_dims(results, 0)
    else:
        results = np.expand_dims(results, 1)

    # If we pass a batch of prompts to the model and ask for
    # several samples we get a list of results that we need
    # to reshape to the right dimensions.
    if len(prompt) > 1 and samples > 1:
        results = np.reshape(results, (-1, samples))

    return results


def create_stop_constraint(
    stop_at: List[str],
    tokenizer: "PreTrainedTokenizerBase",
    prompt_tokens: "torch.Tensor",
) -> Tuple[Callable, Callable, Callable]:
    """Create a constraint that stops generation after a sequence has been found.

    Parameters
    ----------
    stop_at
        The list of sequences which, once generated, the generation is stopped.
    tokenizer
        The tokenizer that corresponds to the model used for generation.
    prompt_tokens
        An array that contains the tokenized prompt.

    """
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]

    def stopping_criterion(input_ids: torch.Tensor, _) -> bool:
        """Choose whether to stop the generation after this step.

        We check whether either of the stopping sequences is present in the
        current generation. If either one is found we stop the generation.

        """
        decoded_input = tokenizer.decode(
            input_ids[0, num_prompt_tokens:], skip_special_tokens=True
        )
        for stopping_sequence in stop_at:
            if stopping_sequence in decoded_input:
                return True

        return False

    def postprocess(generated_sequence: str) -> str:
        """Postprocess the generated text.

        We need to remove the stopping sequence that triggered the end of
        the generation at the end.

        """
        for stopping_sequence in stop_at:
            idx = generated_sequence.find(stopping_sequence)
            if idx != -1:
                return generated_sequence[:idx]

        return generated_sequence

    return lambda _, x: x, stopping_criterion, postprocess


def create_choice_constraint(
    choices: List[str],
    tokenizer: "PreTrainedTokenizerBase",
    prompt_tokens: "torch.Tensor",
) -> Tuple[Callable, Callable, Callable]:
    """Create a constraint that forces the generation to be among a list of choices.

    Parameters
    ----------
    choices
        The list of sequences to which the generated sequences must belong.
    tokenizer
        The tokenizer that corresponds to the model used for generation.
    prompt_tokens
        An array that contains the tokenized prompt.

    """
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]
    tokenized_choices = [tokenizer.encode(word) for word in choices]

    def logit_processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Pre-process the model's output logits before generating the next token.

        At each step we forbid the tokens that do not steer the generation in the
        direction of being either of the choices.

        """
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
        """Choose whether to stop the generation after this step.

        We stop generation when either of the choices has been found.

        TODO: We can stop the generation once we have excluded all possibilities
        but one, and the full sequence can be recovered during post-processing.

        """
        decoded_input = tokenizer.decode(
            input_ids[0, num_prompt_tokens:], skip_special_tokens=True
        )

        is_present_in_output = []
        for choice in choices:
            if choice == decoded_input:
                return True
            elif choice.startswith(decoded_input):
                is_present_in_output.append(1)
            else:
                is_present_in_output.append(0)

        # If we have eliminated all possibilities but one, return
        if sum(is_present_in_output) == 1:
            return True

        return False

    def postprocess(output_sequence: str) -> str:
        for choice in choices:
            if choice.startswith(output_sequence):
                return choice

        return output_sequence

    return logit_processor, stopping_criterion, postprocess


def create_int_constraint(
    tokenizer: "PreTrainedTokenizerBase", prompt_tokens: "torch.Tensor"
) -> Tuple[Callable, Callable, Callable]:
    """Create a constraints that forces the generated sequence to be an integer.

    Parameters
    ----------
    tokenizer
        The tokenizer that corresponds to the model used for generation.
    prompt_tokens
        An array that contains the tokenized prompt.

    """
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]
    mask = create_int_mask(tokenizer.get_vocab())

    def logit_processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Pre-process the model's output logits before generating the next token.

        At each step we forbid the tokens that do not correspond to a digit. We forbid
        EOS tokens until at least one digit has been generated.

        # TODO: Do we need to allow " ", "\n", "\r" and other delimiters?

        """
        if input_ids.shape[1] > num_prompt_tokens + 1:
            mask[tokenizer.eos_token_id] = True
        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")
        return scores

    return logit_processor, lambda *_: False, lambda x: x


def create_float_constraint(
    tokenizer: "PreTrainedTokenizerBase",
    prompt_tokens: "torch.Tensor",
    decimals: int = 3,
) -> Tuple[Callable, Callable, Callable]:
    """Create a constraints that forces the generated sequence to be an floating point number.

    Parameters
    ----------
    tokenizer
        The tokenizer that corresponds to the model used for generation.
    prompt_tokens
        An array that contains the tokenized prompt.

    """
    import torch

    num_prompt_tokens = prompt_tokens.shape[-1]
    mask = create_float_mask(tokenizer.get_vocab())

    def logit_processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Pre-process the model's output logits before generating the next token.

        At each step we forbid the tokens that do not correspond to a digit. We forbid
        EOS tokens until at least one digit has been generated.

        # TODO: Do we need to allow " ", "\n", "\r" and other delimiters?

        """
        if input_ids.shape[1] > num_prompt_tokens + 1:
            mask[tokenizer.eos_token_id] = True
        expanded_mask = mask.expand_as(scores)
        scores[~expanded_mask] = -float("inf")
        return scores

    def stopping_criterion(input_ids: torch.Tensor, _) -> bool:
        """Choose whether to stop the generation after this step.

        We stop generation if the sequence contains more than one period, or
        if the desired number of decimals has been generated.

        """
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

        return False

    def postprocessing(output: str) -> str:
        """Postprocess the generated text.

        We need to remove the trailing period, present if the generation
        was stopped because a second period was found.

        """
        return output.rstrip(".")

    return logit_processor, stopping_criterion, postprocessing


type_to_mask: Dict[str, Callable] = {
    "float": create_float_constraint,
    "int": create_int_constraint,
}


def create_type_constraint(
    type: str, tokenizer: "PreTrainedTokenizerBase", prompt_tokens: "torch.Tensor"
) -> Tuple[Callable, Callable, Callable]:
    if type not in ["int", "float"]:
        raise NotImplementedError(f"Cannot restrict the generation to type {type}")

    return type_to_mask[type](tokenizer, prompt_tokens)
