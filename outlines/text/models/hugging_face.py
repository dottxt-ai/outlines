from outlines.text.models.model import LanguageModel

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "You need to install `transformers` and `torch` to run HuggingFace's Causal LM models."
    )


class HFCausalLM(LanguageModel):
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
    model_id
        The model string identifier in the `transformers` library.
    name
        The name of this `Op` in the graph.

    """

    def __init__(self, model_name: str, name=None):
        """Initialize the GPT2 model.

        We use HuggingFace's Flax implementation of GPT2. This method will download
        the model's weights if they are not yet cached on your machine.

        # TODO: Download the pre-trained weight when the model is executed instead of
        # when the graph is built.

        """

        super().__init__(name=f"HuggingFace {model_name}")
        self.model_name = model_name

    def sample(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            prompt_tokens = prompt_tokens.to("cuda")

        returned_tokens = self.model.generate(
            **prompt_tokens,
            do_sample=True,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] + 1 :]
        new_tokens = new_tokens.squeeze()

        return new_tokens

    def encode(self, sequence: str) -> torch.Tensor:
        """Return a list of token ids from a text sequence.

        Parameters
        ----------
        sequence
            The text sequence to tokenize.

        Returns
        -------
        A dictionary that contains the token ids and the input mask.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer(sequence, return_tensors="pt")

    def decode(self, ids: torch.Tensor) -> str:
        """Return a text sequence from a array of token ids."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)
