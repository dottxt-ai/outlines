import pickle
import warnings
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Union

from outlines.models.base import Model, ModelTypeAdapter
from outlines.models.tokenizer import Tokenizer
from outlines.processors import CFGLogitsProcessor

if TYPE_CHECKING:
    from llama_cpp import Llama


class LlamaCppTokenizer(Tokenizer):
    def __init__(self, model: "Llama"):
        self.eos_token_id = model.token_eos()
        self.eos_token = model.tokenizer().decode([self.eos_token_id])
        self.pad_token_id = self.eos_token_id
        self.special_tokens: Set[str] = set()

        self.vocabulary: Dict[str, int] = dict()

        self.tokenizer = model.tokenizer()

        # TODO: Remove when https://github.com/ggerganov/llama.cpp/pull/5613 is resolved
        self._hf_tokenizer = None
        try:
            self.vocabulary = model.tokenizer_.hf_tokenizer.get_vocab()
            self._hf_tokenizer = model.tokenizer_.hf_tokenizer
        except AttributeError:
            # ###
            for t in range(model.n_vocab()):
                token_piece = model.tokenizer().decode([t])
                self.vocabulary[token_piece] = t

        # ensure stable ordering of vocabulary
        self.vocabulary = {
            tok: tok_id
            for tok, tok_id in sorted(self.vocabulary.items(), key=lambda x: x[1])
        }

        self._hash = None

    def decode(self, token_ids: List[int]) -> List[str]:
        decoded_bytes = self.tokenizer.detokenize(token_ids)
        return [decoded_bytes.decode("utf-8", errors="ignore")]

    def encode(
        self, prompt: Union[str, List[str]], add_bos: bool = True, special: bool = True
    ) -> Tuple[List[int], List[int]]:
        if isinstance(prompt, list):
            raise NotImplementedError(
                "llama-cpp-python tokenizer doesn't support batch tokenization"
            )
        token_ids = self.tokenizer.tokenize(
            prompt.encode("utf-8", errors="ignore"), add_bos=add_bos, special=special
        )
        # generate attention mask, missing from llama-cpp-python
        attention_mask = [
            1 if token_id != self.pad_token_id else 0 for token_id in token_ids
        ]
        return token_ids, attention_mask

    def convert_token_to_string(self, token: str) -> str:
        if self._hf_tokenizer is not None:
            from transformers.file_utils import SPIECE_UNDERLINE

            token_str = self._hf_tokenizer.convert_tokens_to_string([token])
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                token_str = " " + token_str
            return token_str
        else:
            return token

    def __eq__(self, other):
        if not isinstance(other, LlamaCppTokenizer):
            return False
        return self.__getstate__() == other.__getstate__()

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(pickle.dumps(self))
        return self._hash

    def __getstate__(self):
        """Create a stable representation for outlines.caching"""
        return (
            self.vocabulary,
            self.eos_token_id,
            self.eos_token,
            self.pad_token_id,
            sorted(self.special_tokens),
        )

    def __setstate__(self, state):
        raise NotImplementedError("Cannot load a pickled llamacpp tokenizer")


class LlamaCppTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `llama-cpp-python` library.

    `LlamaCppTypeAdapter` is responsible for preparing the arguments to
    `llama-cpp-python`'s `Llama.__call__` method: the input (a string prompt),
    as well as the logits processor (an instance of `LogitsProcessorList`).

    """

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the prompt argument to pass to the model.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        raise NotImplementedError(
            f"The input type {input} is not available. "
            "The `llama-cpp-python` library does not support batch inference. "
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str):
        return model_input

    def format_output_type(self, output_type):
        """Generate the logits processor argument to pass to the model.

        Argument
        --------
        output_type
            The logits processor provided.

        """
        from llama_cpp import LogitsProcessorList

        return LogitsProcessorList([output_type])


class LlamaCpp(Model):
    """Wraps a model provided by the `llama-cpp-python` library."""
    tensor_library_name = "numpy"

    def __init__(self, model: "Llama"):
        from llama_cpp import Llama

        self.model = model
        self.tokenizer = LlamaCppTokenizer(self.model)
        self.type_adapter = LlamaCppTypeAdapter()

    def generate(self, model_input, output_type, **inference_kwargs):
        """Generate text using `llama-cpp-python`.

        Arguments
        ---------
        prompt
            A prompt.
        output_type
            The logits processor to use when generating text.
        inference_kwargs
            The inference kwargs that can be passed to the `Llama.__call__` method
            in the `llama-cpp-python` library.

        Returns
        -------
        The generated text.

        """
        if isinstance(output_type, CFGLogitsProcessor):
            raise NotImplementedError(
                "CFG generation is not supported for LlamaCpp due to bug in the llama_cpp tokenizer"
            )

        completion = self.model(
            self.type_adapter.format_input(model_input),
            logits_processor=self.type_adapter.format_output_type(output_type),
            **inference_kwargs,
        )
        result = completion["choices"][0]["text"]

        self.model.reset()

        return result

    def generate_stream(
        self, model_input, output_type, **inference_kwargs
    ):
        """Stream text using `llama-cpp-python`.

        Arguments
        ---------
        prompt
            A prompt.
        output_type
            The logits processor to use when generating text.
        inference_kwargs
            The inference kwargs that can be passed to the `Llama.__call__` method
            in the `llama-cpp-python` library.

        Returns
        -------
        A generator that return strings.

        """
        if isinstance(output_type, CFGLogitsProcessor):
            raise NotImplementedError(
                "CFG generation is not supported for LlamaCpp due to bug in the llama_cpp tokenizer"
            )

        generator = self.model(
            self.type_adapter.format_input(model_input),
            logits_processor=self.type_adapter.format_output_type(output_type),
            stream=True,
            **inference_kwargs,
        )

        def token_generator() -> Iterator[str]:
            while True:
                try:
                    result = next(generator)
                    yield result["choices"][0]["text"]
                except StopIteration:
                    self.model.reset()
                    return

        return token_generator()


def from_llamacpp(model: "Llama"):
    return LlamaCpp(model)
