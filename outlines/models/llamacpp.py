import pickle
import warnings
from typing import TYPE_CHECKING, Dict, Iterator, List, Set, Tuple, Union

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


class LlamaCpp:
    """Wraps a model provided by the `llama-cpp-python` library."""

    def __init__(self, model_path: Union[str, "Llama"], **kwargs):
        from llama_cpp import Llama

        if isinstance(model_path, Llama):
            self.model = model_path
        else:
            # TODO: Remove when https://github.com/ggerganov/llama.cpp/pull/5613 is resolved
            if "tokenizer" not in kwargs:
                warnings.warn(
                    "The pre-tokenizer in `llama.cpp` handles unicode improperly "
                    + "(https://github.com/ggerganov/llama.cpp/pull/5613)\n"
                    + "Outlines may raise a `RuntimeError` when building the regex index.\n"
                    + "To circumvent this error when using `models.llamacpp()` you may pass the argument"
                    + "`tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(<hf_repo_id>)`\n"
                )

            self.model = Llama(model_path, **kwargs)

        self.tokenizer = LlamaCppTokenizer(self.model)

    @classmethod
    def from_pretrained(cls, repo_id, filename, **kwargs):
        """Download the model weights from Hugging Face and create a `Llama` instance"""
        from llama_cpp import Llama

        # TODO: Remove when https://github.com/ggerganov/llama.cpp/pull/5613 is resolved
        if "tokenizer" not in kwargs:
            warnings.warn(
                "The pre-tokenizer in `llama.cpp` handles unicode improperly "
                + "(https://github.com/ggerganov/llama.cpp/pull/5613)\n"
                + "Outlines may raise a `RuntimeError` when building the regex index.\n"
                + "To circumvent this error when using `models.llamacpp()` you may pass the argument"
                + "`tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(<hf_repo_id>)`\n"
            )

            model = Llama.from_pretrained(repo_id, filename, **kwargs)
            return cls(model)

    def generate(self, prompt: str, logits_processor, **inference_kwargs) -> str:
        """Generate text using `llama-cpp-python`.

        Arguments
        ---------
        prompt
            A prompt.
        logits_processor
            The logits processor to use when generating text.
        inference_kwargs
            The inference kwargs that can be passed to the `Llama.__call__` method
            in the `llama-cpp-python` library.

        Returns
        -------
        The generated text.

        """
        from llama_cpp import LogitsProcessorList

        if not isinstance(prompt, str):
            raise NotImplementedError(
                "The `llama-cpp-python` library does not support batch inference."
            )

        if isinstance(logits_processor, CFGLogitsProcessor):
            raise NotImplementedError(
                "CFG generation is not supported for LlamaCpp due to bug in the llama_cpp tokenizer"
            )

        completion = self.model(
            prompt,
            logits_processor=LogitsProcessorList([logits_processor]),
            **inference_kwargs,
        )
        result = completion["choices"][0]["text"]

        self.model.reset()

        return result

    def stream(
        self, prompt: str, logits_processor, **inference_kwargs
    ) -> Iterator[str]:
        """Stream text using `llama-cpp-python`.

        Arguments
        ---------
        prompt
            A prompt.
        logits_processor
            The logits processor to use when generating text.
        inference_kwargs
            The inference kwargs that can be passed to the `Llama.__call__` method
            in the `llama-cpp-python` library.

        Returns
        -------
        A generator that return strings.

        """
        from llama_cpp import LogitsProcessorList

        if not isinstance(prompt, str):
            raise NotImplementedError(
                "The `llama-cpp-python` library does not support batch inference."
            )

        if isinstance(logits_processor, CFGLogitsProcessor):
            raise NotImplementedError(
                "CFG generation is not supported for LlamaCpp due to bug in the llama_cpp tokenizer"
            )

        generator = self.model(
            prompt,
            logits_processor=LogitsProcessorList([logits_processor]),
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
