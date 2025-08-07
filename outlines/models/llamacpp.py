"""Integration with the `llama-cpp-python` library."""

import ctypes
from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from outlines.inputs import Chat
from outlines.models.base import Model, ModelTypeAdapter
from outlines.models.tokenizer import Tokenizer
from outlines.processors import OutlinesLogitsProcessor

if TYPE_CHECKING:
    from llama_cpp import Llama, LogitsProcessorList

__all__ = ["LlamaCpp", "from_llamacpp"]


class LlamaCppTokenizer(Tokenizer):
    def __init__(self, model: "Llama"):
        self.tokenizer = model.tokenizer()
        self.special_tokens: Set[str] = set()
        self.vocabulary: Dict[str, int] = dict()

        # TODO: Remove when https://github.com/ggerganov/llama.cpp/pull/5613
        # is resolved
        self._hf_tokenizer = None
        if (
            hasattr(model, "tokenizer_")
            and hasattr(model.tokenizer_, "hf_tokenizer")
        ):
            self._hf_tokenizer = model.tokenizer_.hf_tokenizer
            self.eos_token_id = self._hf_tokenizer.eos_token_id
            self.eos_token = self._hf_tokenizer.eos_token
            self.vocabulary = self._hf_tokenizer.get_vocab()
        else:
            from llama_cpp import (
                llama_model_get_vocab,
                llama_token_to_piece,
            )

            self.eos_token_id = model.token_eos()
            size = 32
            buffer = (ctypes.c_char * size)()
            for i in range(model.n_vocab()):
                n = llama_token_to_piece(
                    llama_model_get_vocab(model.model),
                    i,
                    buffer,
                    size,
                    0,
                    True
                )
                token_piece = buffer[:n].decode("utf-8", errors="replace") # type: ignore
                self.vocabulary[token_piece] = i
                if i == self.eos_token_id:
                    self.eos_token = token_piece

        self.pad_token_id = self.eos_token_id
        # ensure stable ordering of vocabulary
        self.vocabulary = {
            tok: tok_id
            for tok, tok_id
            in sorted(self.vocabulary.items(), key=lambda x: x[1])
        }
        self._hash = None

    def decode(self, token_ids: List[int]) -> List[str]:
        decoded_bytes = self.tokenizer.detokenize(token_ids)
        return [decoded_bytes.decode("utf-8", errors="ignore")]

    def encode(
        self,
        prompt: Union[str, List[str]],
        add_bos: bool = True,
        special: bool = True,
    ) -> Tuple[List[int], List[int]]:
        if isinstance(prompt, list):
            raise NotImplementedError(
                "llama-cpp-python tokenizer doesn't support batch tokenization"
            )
        token_ids = self.tokenizer.tokenize(
            prompt.encode("utf-8", errors="ignore"),
            add_bos=add_bos,
            special=special,
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
            if (
                token.startswith(SPIECE_UNDERLINE)
                or token == "<0x20>"
            ):  # pragma: no cover
                token_str = " " + token_str
            return token_str
        else:
            return token

    def __eq__(self, other):
        if not isinstance(other, LlamaCppTokenizer):
            return False
        return self.__getstate__() == other.__getstate__()

    def __hash__(self):
        # We create a custom hash as pickle.dumps(self) is not stable
        if self._hash is None:
            self._hash = hash((
                tuple(sorted(self.vocabulary.items())),
                self.eos_token_id,
                self.eos_token,
                self.pad_token_id,
                tuple(sorted(self.special_tokens)),
            ))
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
    """Type adapter for the `LlamaCpp` model.

    `LlamaCppTypeAdapter` is responsible for preparing the arguments to
    the `Llama` object text generation methods.

    """

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the prompt argument to pass to the model.

        Parameters
        ----------
        model_input
            The input provided by the user.

        Returns
        -------
        str
            The formatted input to be passed to the model.

        """
        raise NotImplementedError(
            f"The input type {type(model_input)} is not available with "
            "LlamaCpp. The only available types are `str` and `Chat`."
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str) -> str:
        return model_input

    @format_input.register(Chat)
    def format_chat_input(self, model_input: Chat) -> list:
        if not all(
            isinstance(message["content"], str)
            for message in model_input.messages
        ):
            raise ValueError(
                "LlamaCpp does not support multi-modal messages."
                + "The content of each message must be a string."
            )

        return  [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in model_input.messages
        ]

    def format_output_type(
        self, output_type: Optional[OutlinesLogitsProcessor] = None,
    ) -> "LogitsProcessorList":
        """Generate the logits processor argument to pass to the model.

        Parameters
        ----------
        output_type
            The logits processor provided.

        Returns
        -------
        LogitsProcessorList
            The logits processor to pass to the model.

        """
        from llama_cpp import LogitsProcessorList

        return LogitsProcessorList([output_type])


class LlamaCpp(Model):
    """Thin wrapper around the `llama_cpp.Llama` model.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `llama_cpp.Llama` model.
    """

    tensor_library_name = "numpy"

    def __init__(self, model: "Llama"):
        """
        Parameters
        ----------
        model
            A `llama_cpp.Llama` model instance.

        """
        self.model = model
        self.tokenizer = LlamaCppTokenizer(self.model)
        self.type_adapter = LlamaCppTypeAdapter()

    def generate(
        self,
        model_input: Union[Chat, str],
        output_type: Optional[OutlinesLogitsProcessor] = None,
        **inference_kwargs: Any,
    ) -> str:
        """Generate text using `llama-cpp-python`.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The logits processor the model will use to constrain the format of
            the generated text.
        **inference_kwargs
            Additional keyword arguments to pass to the `Llama.__call__`
            method of the `llama-cpp-python` library.

        Returns
        -------
        str
            The text generated by the model.

        """
        prompt = self.type_adapter.format_input(model_input)

        if isinstance(prompt, str):
            completion = self.model(
                prompt,
                logits_processor=self.type_adapter.format_output_type(output_type),
                **inference_kwargs,
            )
            result = completion["choices"][0]["text"]
        elif isinstance(prompt, list): # pragma: no cover
            completion = self.model.create_chat_completion(
                prompt,
                logits_processor=self.type_adapter.format_output_type(output_type),
                **inference_kwargs,
            )
            result = completion["choices"][0]["message"]["content"]

        self.model.reset()

        return result

    def generate_batch(
        self,
        model_input,
        output_type = None,
        **inference_kwargs,
    ):
        raise NotImplementedError("LlamaCpp does not support batch generation.")

    def generate_stream(
        self,
        model_input: Union[Chat, str],
        output_type: Optional[OutlinesLogitsProcessor] = None,
        **inference_kwargs: Any,
    ) -> Iterator[str]:
        """Stream text using `llama-cpp-python`.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The logits processor the model will use to constrain the format of
            the generated text.
        **inference_kwargs
            Additional keyword arguments to pass to the `Llama.__call__`
            method of the `llama-cpp-python` library.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        prompt = self.type_adapter.format_input(model_input)

        if isinstance(prompt, str):
            generator = self.model(
                prompt,
                logits_processor=self.type_adapter.format_output_type(output_type),
                stream=True,
                **inference_kwargs,
            )
            for chunk in generator:
                yield chunk["choices"][0]["text"]

        elif isinstance(prompt, list): # pragma: no cover
            generator = self.model.create_chat_completion(
                prompt,
                logits_processor=self.type_adapter.format_output_type(output_type),
                stream=True,
                **inference_kwargs,
            )
            for chunk in generator:
                yield chunk["choices"][0]["delta"].get("content", "")


def from_llamacpp(model: "Llama"):
    """Create an Outlines `LlamaCpp` model instance from a
    `llama_cpp.Llama` instance.

    Parameters
    ----------
    model
        A `llama_cpp.Llama` instance.

    Returns
    -------
    LlamaCpp
        An Outlines `LlamaCpp` model instance.

    """
    return LlamaCpp(model)
