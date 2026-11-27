"""llama-cpp-python passes the sampling parameters to its logits processors as
positional arguments, the Outlines ones take them as keyword arguments. These
tests use a tiny stand-in for `llama_cpp` so they do not need a model file."""

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from outlines.inputs import Chat
from outlines.processors import OutlinesLogitsProcessor


class LlamaLogitsProcessor:
    """Copy of the wrapper `llama_cpp.LogitsProcessorList` builds around a
    processor, see llama_cpp/_logits_processor.py in llama-cpp-python."""

    def __init__(self, logits_processor, **kwargs):
        self.logits_processor = logits_processor
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_ids, scores):
        return self.logits_processor(input_ids, scores)


class LogitsProcessorList(list):
    def __getitem__(self, index):
        processor = list.__getitem__(self, index)
        if isinstance(index, slice):
            return processor
        if isinstance(processor, LlamaLogitsProcessor):
            return processor
        wrapped = LlamaLogitsProcessor(
            processor,
            thought_enabled_kwargs=("skip_prompt", "skip_prompt_stop_str"),
            core_kwargs=("top_k", "top_p", "min_p", "temp", "logit_bias"),
        )
        list.__setitem__(self, index, wrapped)
        return wrapped


class RecordingProcessor(OutlinesLogitsProcessor):
    """Stands in for an Outlines processor, which takes the sampling
    parameters as keyword arguments."""

    def __init__(self):
        super().__init__(tensor_library_name="numpy")
        self.calls: list = []

    def __call__(self, input_ids, logits, **params):
        self.calls.append(params)
        return logits

    def process_logits(self, input_ids, logits):  # pragma: no cover
        return logits


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    module = types.ModuleType("llama_cpp")
    module.LogitsProcessorList = LogitsProcessorList
    module.LlamaLogitsProcessor = LlamaLogitsProcessor
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return module


def _generate(adapter, processor, **inference_kwargs):
    """Run the processor list the way llama-cpp-python does for one token."""
    processor_list = adapter.format_output_type(processor, **inference_kwargs)
    input_ids = np.array([[1, 2, 3]])
    scores = np.full((1, 5), -10.0, dtype=np.float32)
    for item in processor_list:
        scores = item(input_ids, scores)
    return scores


def test_sampling_params_reach_the_processor(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    processor = RecordingProcessor()
    _generate(LlamaCppTypeAdapter(), processor, top_k=5, temp=0.2, max_tokens=32)

    assert len(processor.calls) == 1
    assert processor.calls[0] == {"top_k": 5, "temp": 0.2, "max_tokens": 32}


def test_no_sampling_params(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    processor = RecordingProcessor()
    _generate(LlamaCppTypeAdapter(), processor)

    assert processor.calls == [{}]


def test_stream_is_not_forwarded(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    processor = RecordingProcessor()
    # `stream` is a generation flag, not a sampling parameter, and
    # llama-cpp-python does not pass it to the processors.
    _generate(LlamaCppTypeAdapter(), processor, temp=0.0)

    assert processor.calls[0] == {"temp": 0.0}


def test_no_output_type_returns_none(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    assert LlamaCppTypeAdapter().format_output_type(None) is None


def test_processor_output_is_unchanged(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    guided = _generate(LlamaCppTypeAdapter(), RecordingProcessor(), top_k=1)

    np.testing.assert_array_equal(guided, np.full((1, 5), -10.0, dtype=np.float32))


class FakeTokenizer:
    """Just enough of `llama_cpp.LlamaTokenizer` for `LlamaCppTokenizer`."""

    eos_token_id = 2
    eos_token = "</s>"
    pad_token_id = 2
    special_tokens: ClassVar[set] = set()
    vocabulary: ClassVar[dict] = {"1": 1, "</s>": 2}

    def encode(self, prompt, add_bos=True, special=True):
        return [1]

    def decode(self, token_ids):
        return ["1"]

    def detokenize(self, token_ids):
        return b"1"

    def get_vocab(self):
        return dict(self.vocabulary)

    # Stands for a HF tokenizer, which `LlamaCppTokenizer` prefers because
    # `llama_model_get_vocab` and friends are not always available.
    hf_tokenizer = property(lambda self: self)


class FakeLlama:
    """Calls the processors once per token, the way `llama_cpp.Llama` does."""

    # With `hf_tokenizer` on it, `LlamaCppTokenizer` reads the vocabulary from
    # here instead of the llama.cpp C API.
    tokenizer_ = FakeTokenizer()

    def __init__(self):
        self.tokenizer = FakeTokenizer

    def _run(self, logits_processor):
        input_ids = np.array([[1]])
        scores = np.full((1, 5), -10.0, dtype=np.float32)
        for _ in range(2):
            for index in range(len(logits_processor)):
                logits_processor[index](input_ids, scores)

    def reset(self):
        pass

    def __call__(self, prompt, logits_processor=None, stream=False, **kwargs):
        self._run(logits_processor)
        if stream:
            return iter([{"choices": [{"text": "1"}]}])
        return {"choices": [{"text": "1"}]}

    def create_chat_completion(
        self, prompt, logits_processor=None, stream=False, **kwargs
    ):
        self._run(logits_processor)
        if stream:
            return iter([{"choices": [{"delta": {"content": "1"}}]}])
        return {"choices": [{"message": {"content": "1"}}]}


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_generate_forwards_the_sampling_params(fake_llama_cpp, chat, stream):
    from outlines.models.llamacpp import LlamaCpp

    processor = RecordingProcessor()
    model = LlamaCpp(FakeLlama(), chat_mode=chat)
    model_input = Chat([{"role": "user", "content": "hi"}]) if chat else "hi"

    if stream:
        for _ in model.generate_stream(model_input, processor, top_k=5, temp=0.2):
            pass
    else:
        assert model.generate(model_input, processor, top_k=5, temp=0.2) == "1"

    # Llama runs a processor once per token, on the first one it is called
    # with `(input_ids, scores)` only and the parameters are on the wrapper
    # built by `LogitsProcessorList`.
    assert processor.calls
    assert processor.calls[0] == {"top_k": 5, "temp": 0.2}


def test_generate_without_sampling_params(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCpp

    processor = RecordingProcessor()
    LlamaCpp(FakeLlama(), chat_mode=False).generate("hi", processor)

    assert processor.calls
    assert all(params == {} for params in processor.calls)
