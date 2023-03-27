import random
from typing import Dict

from outlines.text.models.model import LanguageModel

try:
    import jax
    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
except ImportError:
    raise ImportError(
        "You need to install `transformers` and `flax` to run the GTP2 model."
    )


class GPT2(LanguageModel):
    def __init__(self, name=None):
        """Initialize the GPT2 model.

        We use HuggingFace's Flax implementation of GPT2. This method will download
        the model's weights if they are not yet cached on your machine.

        # TODO: Download the pre-trained weight when the model is executed instead of
        # when the graph is built.

        """
        random.seed()
        self.seed = random.randint(0, 2**32)
        super().__init__(name="HuggingFace GPT2")

    def sample(self, prompt_tokens: Dict[str, jax.Array]) -> jax.Array:
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
        self.model = FlaxAutoModelForCausalLM.from_pretrained("gpt2")
        returned_tokens = self.model.generate(
            **prompt_tokens,
            do_sample=True,
            max_new_tokens=20,
            prng_key=jax.random.PRNGKey(self.seed),
            pad_token_id=self.tokenizer.eos_token_id,
        ).sequences
        new_tokens = returned_tokens[:, prompt_tokens["input_ids"].shape[1] + 1 :]
        new_tokens = new_tokens.squeeze()

        return new_tokens

    def encode(self, sequence: str) -> Dict[str, jax.Array]:
        """Return a list of token ids from a text sequence.

        Parameters
        ----------
        sequence
            The text sequence to tokenize.

        Returns
        -------
        A dictionary that contains the token ids and the input mask.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self.tokenizer(sequence, return_tensors="jax")

    def decode(self, ids: jax.Array) -> str:
        """Return a text sequence from a array of token ids."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)
