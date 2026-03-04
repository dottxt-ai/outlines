"""Tests for outlines/exceptions.py and per-provider exception mapping.

Each provider section can be run independently:
    pytest tests/test_exceptions.py -k "openai"
    pytest tests/test_exceptions.py -k "anthropic"
    pytest tests/test_exceptions.py -k "mistral"
    pytest tests/test_exceptions.py -k "gemini"
    pytest tests/test_exceptions.py -k "ollama"
    pytest tests/test_exceptions.py -k "tgi"
    pytest tests/test_exceptions.py -k "dottxt"
    pytest tests/test_exceptions.py -k "vllm"
    pytest tests/test_exceptions.py -k "sglang"
"""

import pytest
from unittest.mock import Mock

from outlines.exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    GenerationError,
    NotFoundError,
    OutlinesError,
    PermissionDeniedError,
    ProviderResponseError,
    RateLimitError,
    ServerError,
    _extract_status_code,
    normalize_provider_exception,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_all_are_outlines_error(self):
        for cls in (
            APIError, AuthenticationError, PermissionDeniedError, NotFoundError,
            RateLimitError, BadRequestError, ServerError, APITimeoutError,
            APIConnectionError, ProviderResponseError, GenerationError,
        ):
            assert issubclass(cls, OutlinesError)

    def test_all_are_api_error(self):
        for cls in (
            AuthenticationError, PermissionDeniedError, NotFoundError,
            RateLimitError, BadRequestError, ServerError, APITimeoutError,
            APIConnectionError, ProviderResponseError, GenerationError,
        ):
            assert issubclass(cls, APIError)

    def test_retryable_flags(self):
        assert RateLimitError.retryable is True
        assert ServerError.retryable is True
        assert APITimeoutError.retryable is True
        assert APIConnectionError.retryable is True
        assert AuthenticationError.retryable is False
        assert PermissionDeniedError.retryable is False
        assert NotFoundError.retryable is False
        assert BadRequestError.retryable is False
        assert ProviderResponseError.retryable is False
        assert GenerationError.retryable is False

    def test_api_error_stores_provider_and_original(self):
        orig = ValueError("boom")
        err = APIError(provider="test", original_exception=orig)
        assert err.provider == "test"
        assert err.original_exception is orig

    def test_api_error_status_code_param(self):
        err = APIError(status_code=429)
        assert err.status_code == 429

    def test_api_error_default_message(self):
        err = APIError()
        assert str(err) == "API error"

    def test_api_error_provider_message(self):
        err = APIError(provider="openai")
        assert "[openai]" in str(err)

    def test_hint_stored_in_notes_or_attribute(self):
        err = AuthenticationError()
        if hasattr(err, "__notes__"):  # Python 3.11+
            assert any("Check API key" in note for note in err.__notes__)
        else:
            assert "Check API key" in err._hint_note

    def test_hint_note_uses_arrow_format(self):
        err = AuthenticationError()
        if hasattr(err, "__notes__"):
            assert any(note.startswith("  →") for note in err.__notes__)
        else:
            assert err._hint_note.startswith("  →")

    def test_hint_not_in_message(self):
        err = AuthenticationError(provider="openai")
        assert "Check API key" not in str(err)
        assert "Check API key" not in err.args[0]

    def test_no_hint_no_notes(self):
        err = APIError()
        assert not getattr(err, "__notes__", [])
        assert not hasattr(err, "_hint_note")

    def test_hint_attribute_accessible(self):
        assert AuthenticationError.hint == "Check API key."
        assert RateLimitError.hint != ""
        assert APIError.hint == ""


# ---------------------------------------------------------------------------
# _extract_status_code
# ---------------------------------------------------------------------------

class TestExtractStatusCode:
    def test_none_input(self):
        assert _extract_status_code(None) is None

    def test_status_code_attr(self):
        class E(Exception):
            status_code = 429
        assert _extract_status_code(E()) == 429

    def test_code_attr(self):
        class E(Exception):
            code = 404
        assert _extract_status_code(E()) == 404

    def test_status_attr(self):
        class E(Exception):
            status = 500
        assert _extract_status_code(E()) == 500

    def test_response_status_code(self):
        class Resp:
            status_code = 403
        class E(Exception):
            response = Resp()
        assert _extract_status_code(E()) == 403

    def test_sentinel_minus_one_filtered(self):
        class E(Exception):
            status_code = -1
        assert _extract_status_code(E()) is None

    def test_non_int_ignored(self):
        class E(Exception):
            status_code = "429"
        assert _extract_status_code(E()) is None


# ---------------------------------------------------------------------------
# normalize_provider_exception — status-code fallback
# (uses provider "test" so no SDK-specific mapping is applied)
# ---------------------------------------------------------------------------

class TestNormalizeProviderException:
    def _exc(self, status_code=None):
        class FakeSDKError(Exception):
            pass
        if status_code is not None:
            FakeSDKError.status_code = status_code
        return FakeSDKError()

    def test_status_code_401(self):
        assert isinstance(normalize_provider_exception(self._exc(401), "test"), AuthenticationError)

    def test_status_code_403(self):
        assert isinstance(normalize_provider_exception(self._exc(403), "test"), PermissionDeniedError)

    def test_status_code_404(self):
        assert isinstance(normalize_provider_exception(self._exc(404), "test"), NotFoundError)

    def test_status_code_429(self):
        assert isinstance(normalize_provider_exception(self._exc(429), "test"), RateLimitError)

    def test_status_code_500(self):
        assert isinstance(normalize_provider_exception(self._exc(500), "test"), ServerError)

    def test_status_code_503(self):
        assert isinstance(normalize_provider_exception(self._exc(503), "test"), ServerError)

    def test_status_code_400(self):
        assert isinstance(normalize_provider_exception(self._exc(400), "test"), BadRequestError)

    def test_status_code_422(self):
        assert isinstance(normalize_provider_exception(self._exc(422), "test"), BadRequestError)

    def test_no_status_code_fallback_to_api_error(self):
        result = normalize_provider_exception(ValueError("unknown"), "test")
        assert type(result) is APIError

    def test_original_exception_preserved(self):
        orig = ValueError("original")
        assert normalize_provider_exception(orig, "test").original_exception is orig

    def test_provider_stored(self):
        result = normalize_provider_exception(self._exc(429), "mycloud")
        assert result.provider == "mycloud"

    def test_unknown_provider_uses_status_code_fallback(self):
        # An unrecognised provider name returns {} from _build_exception_map,
        # so only the status-code fallback applies.
        assert isinstance(normalize_provider_exception(self._exc(429), "unknown-provider"), RateLimitError)


# ---------------------------------------------------------------------------
# OpenAI provider mapping
# ---------------------------------------------------------------------------

class TestOpenAIExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip("openai")

    def _check(self, sdk_exc_cls, expected_outlines_cls):
        result = normalize_provider_exception(Mock(spec=sdk_exc_cls), "openai")
        assert isinstance(result, expected_outlines_cls), (
            f"Expected {expected_outlines_cls.__name__}, got {type(result).__name__}"
        )

    def test_authentication_error(self):
        import openai
        self._check(openai.AuthenticationError, AuthenticationError)

    def test_permission_denied(self):
        import openai
        self._check(openai.PermissionDeniedError, PermissionDeniedError)

    def test_not_found(self):
        import openai
        self._check(openai.NotFoundError, NotFoundError)

    def test_rate_limit(self):
        import openai
        self._check(openai.RateLimitError, RateLimitError)

    def test_bad_request(self):
        import openai
        self._check(openai.BadRequestError, BadRequestError)

    def test_internal_server_error(self):
        import openai
        self._check(openai.InternalServerError, ServerError)

    def test_timeout(self):
        import openai
        self._check(openai.APITimeoutError, APITimeoutError)

    def test_connection_error(self):
        import openai
        self._check(openai.APIConnectionError, APIConnectionError)

    def test_response_validation(self):
        import openai
        self._check(openai.APIResponseValidationError, ProviderResponseError)

    def test_length_finish_reason(self):
        import openai
        self._check(openai.LengthFinishReasonError, GenerationError)

    def test_content_filter_finish_reason(self):
        import openai
        self._check(openai.ContentFilterFinishReasonError, GenerationError)


# ---------------------------------------------------------------------------
# Anthropic provider mapping
# ---------------------------------------------------------------------------

class TestAnthropicExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_anthropic(self):
        pytest.importorskip("anthropic")

    def _check(self, sdk_exc_cls, expected_outlines_cls):
        result = normalize_provider_exception(Mock(spec=sdk_exc_cls), "anthropic")
        assert isinstance(result, expected_outlines_cls)

    def test_authentication_error(self):
        import anthropic
        self._check(anthropic.AuthenticationError, AuthenticationError)

    def test_permission_denied(self):
        import anthropic
        self._check(anthropic.PermissionDeniedError, PermissionDeniedError)

    def test_not_found(self):
        import anthropic
        self._check(anthropic.NotFoundError, NotFoundError)

    def test_rate_limit(self):
        import anthropic
        self._check(anthropic.RateLimitError, RateLimitError)

    def test_bad_request(self):
        import anthropic
        self._check(anthropic.BadRequestError, BadRequestError)

    def test_overloaded(self):
        import anthropic._exceptions as anthropic_exc
        self._check(anthropic_exc.OverloadedError, ServerError)

    def test_service_unavailable(self):
        import anthropic._exceptions as anthropic_exc
        self._check(anthropic_exc.ServiceUnavailableError, ServerError)

    def test_timeout(self):
        import anthropic
        self._check(anthropic.APITimeoutError, APITimeoutError)

    def test_connection_error(self):
        import anthropic
        self._check(anthropic.APIConnectionError, APIConnectionError)

    def test_response_validation(self):
        import anthropic
        self._check(anthropic.APIResponseValidationError, ProviderResponseError)


# ---------------------------------------------------------------------------
# Mistral provider mapping
# ---------------------------------------------------------------------------

class TestMistralExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_mistral(self):
        pytest.importorskip("mistralai")

    def _check(self, sdk_exc_cls, expected_outlines_cls):
        result = normalize_provider_exception(Mock(spec=sdk_exc_cls), "mistral")
        assert isinstance(result, expected_outlines_cls)

    def test_http_validation_error(self):
        import mistralai.models as mm
        self._check(mm.HTTPValidationError, BadRequestError)

    def test_validation_error(self):
        import mistralai.models as mm
        self._check(mm.ValidationError, BadRequestError)

    def test_response_validation_error(self):
        import mistralai.models as mm
        self._check(mm.ResponseValidationError, ProviderResponseError)

    def test_no_response_error(self):
        import mistralai.models as mm
        self._check(mm.NoResponseError, APIConnectionError)

    def test_httpx_timeout(self):
        import httpx
        self._check(httpx.TimeoutException, APITimeoutError)

    def test_httpx_connect_error(self):
        import httpx
        self._check(httpx.ConnectError, APIConnectionError)

    def test_status_code_fallback_401(self):
        class MistralSDKError(Exception):
            status_code = 401
        assert isinstance(normalize_provider_exception(MistralSDKError(), "mistral"), AuthenticationError)

    def test_status_code_fallback_429(self):
        class MistralSDKError(Exception):
            status_code = 429
        assert isinstance(normalize_provider_exception(MistralSDKError(), "mistral"), RateLimitError)

    def test_status_code_fallback_500(self):
        class MistralSDKError(Exception):
            status_code = 500
        assert isinstance(normalize_provider_exception(MistralSDKError(), "mistral"), ServerError)


# ---------------------------------------------------------------------------
# Gemini provider mapping
# ---------------------------------------------------------------------------

class TestGeminiExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_gemini(self):
        pytest.importorskip("google.genai")
        pytest.importorskip("aiohttp")

    def _check(self, sdk_exc_cls, expected_outlines_cls):
        result = normalize_provider_exception(Mock(spec=sdk_exc_cls), "gemini")
        assert isinstance(result, expected_outlines_cls)

    def test_server_error(self):
        from google.genai import errors as genai_errors
        self._check(genai_errors.ServerError, ServerError)

    def test_httpx_timeout(self):
        import httpx
        self._check(httpx.TimeoutException, APITimeoutError)

    def test_httpx_connect_error(self):
        import httpx
        self._check(httpx.ConnectError, APIConnectionError)

    def test_aiohttp_server_timeout(self):
        import aiohttp
        self._check(aiohttp.ServerTimeoutError, APITimeoutError)

    def test_aiohttp_client_connector_error(self):
        import aiohttp
        self._check(aiohttp.ClientConnectorError, APIConnectionError)

    def test_client_error_status_code_fallback(self):
        from google.genai import errors as genai_errors
        mock_exc = Mock(spec=genai_errors.ClientError)
        mock_exc.code = 401
        result = normalize_provider_exception(mock_exc, "gemini")
        assert isinstance(result, AuthenticationError)


# ---------------------------------------------------------------------------
# Ollama provider mapping
# ---------------------------------------------------------------------------

class TestOllamaExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_ollama(self):
        pytest.importorskip("ollama")

    def test_request_error(self):
        import ollama
        result = normalize_provider_exception(ollama.RequestError("msg"), "ollama")
        assert isinstance(result, APIConnectionError)

    def test_response_error_with_status_code(self):
        import ollama
        result = normalize_provider_exception(ollama.ResponseError("not found", status_code=404), "ollama")
        assert isinstance(result, NotFoundError)

    def test_response_error_sentinel_minus_one(self):
        import ollama
        result = normalize_provider_exception(ollama.ResponseError("unknown", status_code=-1), "ollama")
        assert type(result) is APIError

    def test_response_error_429(self):
        import ollama
        result = normalize_provider_exception(ollama.ResponseError("rate limited", status_code=429), "ollama")
        assert isinstance(result, RateLimitError)


# ---------------------------------------------------------------------------
# TGI provider mapping
# ---------------------------------------------------------------------------

class TestTGIExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_tgi(self):
        pytest.importorskip("huggingface_hub")

    def _check(self, sdk_exc_cls, expected_outlines_cls):
        result = normalize_provider_exception(Mock(spec=sdk_exc_cls), "tgi")
        assert isinstance(result, expected_outlines_cls)

    def test_inference_timeout(self):
        from huggingface_hub.errors import InferenceTimeoutError
        self._check(InferenceTimeoutError, APITimeoutError)

    def test_overloaded(self):
        from huggingface_hub.errors import OverloadedError
        self._check(OverloadedError, ServerError)

    def test_validation_error(self):
        from huggingface_hub.errors import ValidationError
        self._check(ValidationError, BadRequestError)

    def test_generation_error(self):
        from huggingface_hub.errors import GenerationError as HFGenerationError
        self._check(HFGenerationError, GenerationError)

    def test_incomplete_generation_error(self):
        from huggingface_hub.errors import IncompleteGenerationError
        self._check(IncompleteGenerationError, GenerationError)

    def test_bad_request_error(self):
        from huggingface_hub.errors import BadRequestError as HFBadRequestError
        self._check(HFBadRequestError, BadRequestError)

    def test_gated_repo_error(self):
        from huggingface_hub.errors import GatedRepoError
        self._check(GatedRepoError, PermissionDeniedError)

    def test_repository_not_found(self):
        from huggingface_hub.errors import RepositoryNotFoundError
        self._check(RepositoryNotFoundError, NotFoundError)


# ---------------------------------------------------------------------------
# Dottxt provider mapping
# ---------------------------------------------------------------------------

class TestDottxtExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_dottxt(self):
        pytest.importorskip("urllib3")

    def _check(self, sdk_exc_cls, expected_outlines_cls):
        result = normalize_provider_exception(Mock(spec=sdk_exc_cls), "dottxt")
        assert isinstance(result, expected_outlines_cls)

    def test_connect_timeout(self):
        from urllib3.exceptions import ConnectTimeoutError
        self._check(ConnectTimeoutError, APITimeoutError)

    def test_read_timeout(self):
        from urllib3.exceptions import ReadTimeoutError
        self._check(ReadTimeoutError, APITimeoutError)

    def test_max_retry_error(self):
        from urllib3.exceptions import MaxRetryError
        self._check(MaxRetryError, APIConnectionError)

    def test_new_connection_error(self):
        from urllib3.exceptions import NewConnectionError
        self._check(NewConnectionError, APIConnectionError)


# ---------------------------------------------------------------------------
# vLLM provider mapping (reuses OpenAI map via provider name)
# ---------------------------------------------------------------------------

class TestVLLMExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip("openai")

    def test_authentication_error(self):
        import openai
        result = normalize_provider_exception(Mock(spec=openai.AuthenticationError), "vllm")
        assert isinstance(result, AuthenticationError)

    def test_rate_limit(self):
        import openai
        result = normalize_provider_exception(Mock(spec=openai.RateLimitError), "vllm")
        assert isinstance(result, RateLimitError)

    def test_provider_name(self):
        import openai
        result = normalize_provider_exception(Mock(spec=openai.RateLimitError), "vllm")
        assert result.provider == "vllm"


# ---------------------------------------------------------------------------
# SGLang provider mapping (reuses OpenAI map via provider name)
# ---------------------------------------------------------------------------

class TestSGLangExceptionMap:
    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip("openai")

    def test_authentication_error(self):
        import openai
        result = normalize_provider_exception(Mock(spec=openai.AuthenticationError), "sglang")
        assert isinstance(result, AuthenticationError)

    def test_rate_limit(self):
        import openai
        result = normalize_provider_exception(Mock(spec=openai.RateLimitError), "sglang")
        assert isinstance(result, RateLimitError)

    def test_provider_name(self):
        import openai
        result = normalize_provider_exception(Mock(spec=openai.RateLimitError), "sglang")
        assert result.provider == "sglang"
