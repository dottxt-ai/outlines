from typing import Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "You need to install `transformers` and `torch` to run HuggingFace's Causal LM models."
    )


class HFCausalLM:
    """Represent any of HuggingFace's causal language model implementations.

    You should have the `torch` and `transformers` packages installed. First
    execution may take a while since the pre-trained weights will be downloaded.

    Available models are listed on https://huggingface.co/models

    Example
    ------

    >> from outlines.text.models import HFCausalLM
    >> from outlines.text import string
    >>
    >> gpt2 = HFCausalLM("gpt2")
    >> in = string()
    >> out = gpt2(in)

    Attributes
    ----------
    model_name
        The model string identifier in the `transformers` library.

    """

    def __init__(
        self,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """Instantiate the model `Op`.

        Parameters
        ----------
        model
            The model id of a model hosted inside a model repo on huggingface.co

        """
        if max_tokens is None:
            max_tokens = 216
        self.max_tokens = max_tokens

        if temperature is None:
            temperature = 1.0
        self.temperature = temperature

        self.model_name = model

    def __call__(self, prompt: str) -> str:
        """Sample new tokens give the tokenized prompt.

        Since HuggingFace's `generate` method returns the prompt along with the
        generated token we need to truncate the returned array of tokens.

        Parameters
        ----------
        prompt_tokens
            A dictionary that contains the ids of the tokens contained in the input
            prompt and the input mask. This is the default output of HuggingFace's
            tokenizers.

        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        prompt_tokens = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            model = model.to("cuda")
            prompt_tokens = prompt_tokens.to("cuda")

        returned_tokens = model.generate(
            **prompt_tokens,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] + 1 :]
        new_tokens = new_tokens.squeeze()

        return tokenizer.decode(new_tokens, skip_special_tokens=True)
