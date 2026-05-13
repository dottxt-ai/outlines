from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from outlines.exceptions import BadRequestError, GenerationError, RateLimitError
from outlines.models.anthropic import Anthropic
from outlines.models.dottxt import AsyncDottxt, Dottxt
from outlines.models.gemini import Gemini
from outlines.models.mistral import AsyncMistral, Mistral
from outlines.models.ollama import AsyncOllama, Ollama
from outlines.models.openai import AsyncOpenAI, OpenAI
from outlines.models.sglang import AsyncSGLang, SGLang
from outlines.models.tgi import AsyncTGI, TGI
from outlines.models.vllm import AsyncVLLM, VLLM


class StatusCodeProviderError(Exception):
    # One status code is enough here: these tests verify each wrapper's
    # catch/normalize plumbing. tests/test_exceptions.py covers the full
    # status-code mapping; 429 gives a specific RateLimitError assertion.
    status_code = 429


class RaisingIterator:
    def __init__(self, exc):
        self.exc = exc

    def __iter__(self):
        return self

    def __next__(self):
        raise self.exc


class RaisingAsyncIterator:
    def __init__(self, exc):
        self.exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self.exc


@dataclass(frozen=True)
class ProviderCase:
    provider: str
    make_model: object
    make_client: object
    invoke: object


@dataclass
class ProviderDef:
    """Per-provider config; derives the four pytest case lists automatically."""

    name: str
    model_sync: object = None
    model_async: object = None
    generate_sync: object = None          # sync generate client builder
    stream_sync: object = None            # sync stream  (defaults to generate_sync)
    generate_async: object = None         # async generate (defaults to generate_sync)
    stream_async: object = None           # async stream  (defaults to stream_sync)
    generate_invoke: object = None        # generate call  (defaults to m.generate("hello"))
    has_stream: bool = True
    has_refusal: bool = False

    def __post_init__(self):
        if self.stream_sync is None:
            self.stream_sync = self.generate_sync
        if self.generate_async is None:
            self.generate_async = self.generate_sync
        if self.stream_async is None:
            self.stream_async = self.stream_sync
        if self.generate_invoke is None:
            self.generate_invoke = lambda m: m.generate("hello")

    def sync_generate_case(self):
        return pytest.param(
            ProviderCase(self.name, self.model_sync, self.generate_sync, self.generate_invoke),
            id=self.name,
        )

    def sync_stream_case(self):
        return pytest.param(
            ProviderCase(self.name, self.model_sync, self.stream_sync, lambda m: next(m.stream("hello"))),
            id=self.name,
        )

    def async_generate_case(self):
        return pytest.param(
            ProviderCase(self.name, self.model_async, self.generate_async, self.generate_invoke),
            id=self.name,
        )

    def async_stream_case(self):
        return pytest.param(
            ProviderCase(self.name, self.model_async, self.stream_async, lambda m: _anext(m.stream("hello"))),
            id=self.name,
        )


# OpenAI-compatible
def _chat_completions(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

# Anthropic
def _anthropic(create):
    return SimpleNamespace(messages=SimpleNamespace(create=create))

# Gemini
def _gemini_generate(fn):
    return SimpleNamespace(models=SimpleNamespace(generate_content=fn))

def _gemini_stream(fn):
    return SimpleNamespace(models=SimpleNamespace(generate_content_stream=fn))

# Mistral
def _mistral_complete(fn):
    return SimpleNamespace(chat=SimpleNamespace(complete=fn))

def _mistral_stream(fn):
    return SimpleNamespace(chat=SimpleNamespace(stream=fn))

def _mistral_async_complete(fn):
    return SimpleNamespace(chat=SimpleNamespace(complete_async=fn))

def _mistral_stream_async(fn):
    return SimpleNamespace(chat=SimpleNamespace(stream_async=fn))

# Others
def _ollama(fn):
    return SimpleNamespace(chat=fn)

def _tgi(fn):
    return SimpleNamespace(text_generation=fn)

def _dottxt(fn):
    return SimpleNamespace(generate=fn)


async def _anext(async_stream):
    return await async_stream.__anext__()


def _refusal_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="safety refusal"))]
    )


class DottxtSchema(BaseModel):
    value: str


PROVIDERS = [
    ProviderDef("anthropic", model_sync=Anthropic, generate_sync=_anthropic),
    ProviderDef("gemini", model_sync=Gemini, generate_sync=_gemini_generate, stream_sync=_gemini_stream),
    ProviderDef("openai", model_sync=OpenAI, model_async=AsyncOpenAI, generate_sync=_chat_completions, has_refusal=True),
    ProviderDef(
        "mistral",
        model_sync=Mistral,
        model_async=AsyncMistral,
        generate_sync=_mistral_complete,
        stream_sync=_mistral_stream,
        generate_async=_mistral_async_complete,
        stream_async=_mistral_stream_async,
    ),
    ProviderDef("ollama", model_sync=Ollama, model_async=AsyncOllama, generate_sync=_ollama),
    ProviderDef("tgi", model_sync=TGI, model_async=AsyncTGI, generate_sync=_tgi),
    ProviderDef(
        "dottxt",
        model_sync=lambda client: Dottxt(client, "test-model"),
        model_async=lambda client: AsyncDottxt(client, "test-model"),
        generate_sync=_dottxt,
        generate_invoke=lambda m: m.generate("hello", DottxtSchema),
        has_stream=False,
    ),
    ProviderDef("vllm", model_sync=VLLM, model_async=AsyncVLLM, generate_sync=_chat_completions, has_refusal=True),
    ProviderDef("sglang", model_sync=SGLang, model_async=AsyncSGLang, generate_sync=_chat_completions, has_refusal=True),
]

# Asymmetry across these case lists is driven by what each provider/wrapper
# actually supports:
#   * Dottxt is excluded from all *_STREAM_CASES — its `generate_stream`
#     unconditionally raises NotImplementedError (see models/dottxt.py).
#   * Anthropic and Gemini are excluded from all ASYNC_* cases — they only
#     ship sync model classes (no AsyncAnthropic / AsyncGemini).
SYNC_GENERATE_CASES  = [p.sync_generate_case()  for p in PROVIDERS if p.model_sync]
SYNC_STREAM_CASES    = [p.sync_stream_case()     for p in PROVIDERS if p.model_sync  and p.has_stream]
ASYNC_GENERATE_CASES = [p.async_generate_case()  for p in PROVIDERS if p.model_async]
ASYNC_STREAM_CASES   = [p.async_stream_case()    for p in PROVIDERS if p.model_async and p.has_stream]

SYNC_REFUSAL_CASES  = [pytest.param(p.model_sync,  p.name, id=p.name) for p in PROVIDERS if p.has_refusal and p.model_sync]
ASYNC_REFUSAL_CASES = [pytest.param(p.model_async, p.name, id=p.name) for p in PROVIDERS if p.has_refusal and p.model_async]


@pytest.mark.parametrize("case", SYNC_GENERATE_CASES)
def test_normalize_provider_errors_generate_sync(case):
    original = StatusCodeProviderError("rate limited")
    client = case.make_client(Mock(side_effect=original))
    model = case.make_model(client)

    with pytest.raises(RateLimitError) as exc_info:
        case.invoke(model)

    assert exc_info.value.provider == case.provider
    assert exc_info.value.original_exception is original


@pytest.mark.parametrize("case", SYNC_STREAM_CASES)
def test_normalize_provider_errors_stream_sync(case):
    original = StatusCodeProviderError("rate limited")
    client = case.make_client(Mock(return_value=RaisingIterator(original)))
    model = case.make_model(client)

    with pytest.raises(RateLimitError) as exc_info:
        case.invoke(model)

    assert exc_info.value.provider == case.provider
    assert exc_info.value.original_exception is original


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ASYNC_GENERATE_CASES)
async def test_normalize_provider_errors_generate_async(case):
    original = StatusCodeProviderError("rate limited")
    client = case.make_client(AsyncMock(side_effect=original))
    model = case.make_model(client)

    with pytest.raises(RateLimitError) as exc_info:
        await case.invoke(model)

    assert exc_info.value.provider == case.provider
    assert exc_info.value.original_exception is original


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ASYNC_STREAM_CASES)
async def test_normalize_provider_errors_stream_async(case):
    original = StatusCodeProviderError("rate limited")
    client = case.make_client(AsyncMock(return_value=RaisingAsyncIterator(original)))
    model = case.make_model(client)

    with pytest.raises(RateLimitError) as exc_info:
        await case.invoke(model)

    assert exc_info.value.provider == case.provider
    assert exc_info.value.original_exception is original


@pytest.mark.parametrize("case", SYNC_GENERATE_CASES)
def test_pass_through_non_provider_errors_generate_sync(case):
    original = TypeError("programmer error")
    client = case.make_client(Mock(side_effect=original))
    model = case.make_model(client)

    with pytest.raises(TypeError) as exc_info:
        case.invoke(model)

    assert exc_info.value is original


@pytest.mark.parametrize("case", SYNC_STREAM_CASES)
def test_pass_through_non_provider_errors_stream_sync(case):
    original = TypeError("programmer error")
    client = case.make_client(Mock(return_value=RaisingIterator(original)))
    model = case.make_model(client)

    with pytest.raises(TypeError) as exc_info:
        case.invoke(model)

    assert exc_info.value is original


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ASYNC_GENERATE_CASES)
async def test_pass_through_non_provider_errors_generate_async(case):
    original = TypeError("programmer error")
    client = case.make_client(AsyncMock(side_effect=original))
    model = case.make_model(client)

    with pytest.raises(TypeError) as exc_info:
        await case.invoke(model)

    assert exc_info.value is original


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ASYNC_STREAM_CASES)
async def test_pass_through_non_provider_errors_stream_async(case):
    original = TypeError("programmer error")
    client = case.make_client(AsyncMock(return_value=RaisingAsyncIterator(original)))
    model = case.make_model(client)

    with pytest.raises(TypeError) as exc_info:
        await case.invoke(model)

    assert exc_info.value is original


@pytest.mark.parametrize("model_cls, provider", SYNC_REFUSAL_CASES)
def test_refusal_raises_generation_error_sync(model_cls, provider):
    client = _chat_completions(Mock(return_value=_refusal_response()))
    model = model_cls(client)

    with pytest.raises(GenerationError, match="refused to answer the request") as exc_info:
        model.generate("hello")

    assert exc_info.value.provider == provider


@pytest.mark.asyncio
@pytest.mark.parametrize("model_cls, provider", ASYNC_REFUSAL_CASES)
async def test_refusal_raises_generation_error_async(model_cls, provider):
    client = _chat_completions(AsyncMock(return_value=_refusal_response()))
    model = model_cls(client)

    with pytest.raises(GenerationError, match="refused to answer the request") as exc_info:
        await model.generate("hello")

    assert exc_info.value.provider == provider


# ---------------------------------------------------------------------------
# End-to-end: a real openai.BadRequestError flows through the wrapper and
# surfaces as outlines.exceptions.BadRequestError.
# ---------------------------------------------------------------------------

def test_openai_bad_request_normalizes():
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    err = openai.BadRequestError("Bad request", response=response, body=None)

    client = _chat_completions(Mock(side_effect=err))
    model = OpenAI(client)

    with pytest.raises(BadRequestError):
        model.generate("hello")


def test_mistral_bad_request_normalizes():
    import httpx
    from mistralai.models.sdkerror import SDKError

    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    err = SDKError("API error occurred", response, '{"detail":"bad request"}')

    client = _mistral_complete(Mock(side_effect=err))
    model = Mistral(client)

    with pytest.raises(BadRequestError):
        model.generate("hello")
