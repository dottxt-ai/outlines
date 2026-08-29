"""Tests for outlines/exceptions.py and per-provider exception mapping.

Each provider section can be run independently:
    pytest tests/test_exceptions.py -k "openai"
    pytest tests/test_exceptions.py -k "anthropic"
    pytest tests/test_exceptions.py -k "mistral"
    pytest tests/test_exceptions.py -k "gemini"
    pytest tests/test_exceptions.py -k "ollama"
    pytest tests/test_exceptions.py -k "tgi"
    pytest tests/test_exceptions.py -k "dottxt"
    pytest tests/test_exceptions.py -k "lmstudio"
    pytest tests/test_exceptions.py -k "vllm"
    pytest tests/test_exceptions.py -k "sglang"
"""

import pytest
import sys
from types import ModuleType
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
    _build_exception_map,
    _extract_request_id,
    _extract_status_code,
    is_provider_exception,
    normalize_provider_errors,
    normalize_provider_exception,
)


def fake_provider_error(status_code=None):
    class FakeSDKError(Exception):
        pass

    if status_code is not None:
        FakeSDKError.status_code = status_code
    return FakeSDKError()


def _check_mapping(sdk_exc_cls, provider, expected_outlines_cls):
    """Assert that a mocked SDK exception normalizes to the expected Outlines class."""
    result = normalize_provider_exception(Mock(spec=sdk_exc_cls), provider)
    assert isinstance(result, expected_outlines_cls), (
        f"Expected {expected_outlines_cls.__name__}, got {type(result).__name__}"
    )


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

    def test_hint_note_fallback_without_add_note(self):
        class LegacyAuthenticationError(AuthenticationError):
            def __getattribute__(self, name):
                if name == "add_note":
                    raise AttributeError
                return super().__getattribute__(name)

        err = LegacyAuthenticationError()
        assert err._hint_note == "  → Check API key."

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


class TestExtractStatusCode:
    def test_none_input(self):
        assert _extract_status_code(None) is None

    def test_status_code_attr(self):
        assert _extract_status_code(fake_provider_error(429)) == 429

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

    def test_response_with_non_int_status_code_returns_none(self):
        class Resp:
            status_code = "200"
        class E(Exception):
            response = Resp()
        assert _extract_status_code(E()) is None

    def test_response_with_out_of_range_status_code_returns_none(self):
        class Resp:
            status_code = 99
        class E(Exception):
            response = Resp()
        assert _extract_status_code(E()) is None

    def test_value_600_out_of_range(self):
        assert _extract_status_code(fake_provider_error(600)) is None


class TestNormalizeProviderException:
    @pytest.mark.parametrize(
        ("status_code", "expected_cls"),
        [
            (401, AuthenticationError),
            (403, PermissionDeniedError),
            (404, NotFoundError),
            (429, RateLimitError),
            (400, BadRequestError),
            (409, BadRequestError),
            (413, BadRequestError),
            (418, BadRequestError),
            (422, BadRequestError),
            (500, ServerError),
            (503, ServerError),
        ],
    )
    def test_status_code_fallbacks(self, status_code, expected_cls):
        result = normalize_provider_exception(
            fake_provider_error(status_code),
            "test",
        )
        assert isinstance(result, expected_cls)

    def test_2xx_status_code_falls_back_to_api_error(self):
        result = normalize_provider_exception(fake_provider_error(200), "test")
        assert type(result) is APIError

    def test_no_status_code_fallback_to_api_error(self):
        result = normalize_provider_exception(ValueError("unknown"), "test")
        assert type(result) is APIError

    def test_original_exception_preserved(self):
        orig = ValueError("original")
        assert normalize_provider_exception(orig, "test").original_exception is orig

    def test_provider_stored(self):
        result = normalize_provider_exception(fake_provider_error(429), "mycloud")
        assert result.provider == "mycloud"

    def test_unknown_provider_uses_status_code_fallback(self):
        # An unrecognised provider name returns {} from _build_exception_map,
        # so only the status-code fallback applies.
        result = normalize_provider_exception(
            fake_provider_error(429),
            "unknown-provider",
        )
        assert isinstance(result, RateLimitError)

    def test_request_id_propagated_through_normalize(self):
        # request_id from the original SDK exception should end up on the
        # normalized Outlines exception.
        class FakeSDKError(Exception):
            status_code = 500
            request_id = "req-xyz"

        err = normalize_provider_exception(FakeSDKError(), "test")
        assert err.request_id == "req-xyz"

    def test_request_id_from_response_headers_propagated(self):
        class Resp:
            status_code = 500
            headers = {"x-request-id": "hdr-abc"}

        class FakeSDKError(Exception):
            response = Resp()

        err = normalize_provider_exception(FakeSDKError(), "test")
        assert err.request_id == "hdr-abc"


class TestIsProviderException:
    """is_provider_exception must return False for programmer errors and True
    for genuine provider/transport exceptions."""

    # --- programmer errors must pass through ---

    def test_type_error_is_not_provider(self):
        assert not is_provider_exception(TypeError("bad arg"), "openai")

    def test_attribute_error_is_not_provider(self):
        assert not is_provider_exception(AttributeError("no attr"), "openai")

    def test_key_error_is_not_provider(self):
        assert not is_provider_exception(KeyError("missing"), "openai")

    def test_name_error_is_not_provider(self):
        assert not is_provider_exception(NameError("undef"), "openai")

    def test_assertion_error_is_not_provider(self):
        assert not is_provider_exception(AssertionError("assert"), "openai")

    def test_plain_exception_no_status_code_is_not_provider(self):
        assert not is_provider_exception(Exception("generic"), "openai")

    # --- exceptions with HTTP status codes are treated as provider errors ---

    def test_status_code_429_is_provider(self):
        assert is_provider_exception(fake_provider_error(429), "openai")

    def test_status_code_500_is_provider(self):
        assert is_provider_exception(fake_provider_error(500), "unknown-provider")

    def test_status_code_on_response_is_provider(self):
        class Resp:
            status_code = 403
        class FakeSDKError(Exception):
            response = Resp()
        assert is_provider_exception(FakeSDKError(), "openai")

    def test_sentinel_minus_one_is_not_provider(self):
        # ollama.ResponseError uses -1 when status is unknown; should not wrap
        class FakeSDKError(Exception):
            status_code = -1
        assert not is_provider_exception(FakeSDKError(), "ollama")

    # --- known SDK exception classes are caught by the explicit map ---

    def test_openai_sdk_exception_is_provider(self):
        openai = pytest.importorskip("openai")
        assert is_provider_exception(Mock(spec=openai.RateLimitError), "openai")

    def test_anthropic_sdk_exception_is_provider(self):
        anthropic = pytest.importorskip("anthropic")
        assert is_provider_exception(Mock(spec=anthropic.APITimeoutError), "anthropic")

    # --- unknown provider with no status code falls through ---

    def test_unknown_provider_no_status_code_is_not_provider(self):
        assert not is_provider_exception(ValueError("oops"), "unknown-provider")


class TestProviderSDKNotInstalled:
    """Exception normalization must not require the provider's optional SDK.

    ``is_provider_exception`` / ``normalize_provider_exception`` are public API
    and their contract is to let non-provider errors pass through and to fall
    back to HTTP status-code inspection. Building the SDK-exception map must
    therefore degrade gracefully to an empty map when the provider package is
    absent, instead of raising ``ImportError`` (which would mask the original
    exception when a *different* provider's SDK is the one installed).

    Each provider's optional import is blocked via ``sys.modules[name] = None``,
    which forces ``import name`` to raise ``ImportError`` even if the package is
    installed, so these tests are deterministic regardless of the test env.
    """

    # provider -> the top-level module(s) its exception map imports
    _PROVIDER_SDK_MODULES = {
        "openai": ["openai"],
        "vllm": ["openai"],
        "sglang": ["openai"],
        "anthropic": ["anthropic"],
        "gemini": ["google", "httpx"],
        "ollama": ["ollama", "httpx"],
        "tgi": ["huggingface_hub"],
        "dottxt": ["urllib3"],
    }

    @pytest.fixture
    def block_sdk(self, monkeypatch):
        def _block(*module_names):
            _build_exception_map.cache_clear()
            for name in module_names:
                monkeypatch.setitem(sys.modules, name, None)
        yield _block
        _build_exception_map.cache_clear()

    @pytest.mark.parametrize("provider", sorted(_PROVIDER_SDK_MODULES))
    def test_build_exception_map_returns_empty_when_sdk_missing(self, provider, block_sdk):
        block_sdk(*self._PROVIDER_SDK_MODULES[provider])
        assert _build_exception_map(provider) == {}

    @pytest.mark.parametrize("provider", sorted(_PROVIDER_SDK_MODULES))
    def test_is_provider_exception_passes_through_when_sdk_missing(self, provider, block_sdk):
        block_sdk(*self._PROVIDER_SDK_MODULES[provider])
        # A programmer error must not be misreported as a provider error, and
        # building the (unavailable) SDK map must not raise ImportError.
        assert is_provider_exception(TypeError("programmer bug"), provider) is False

    @pytest.mark.parametrize("provider", sorted(_PROVIDER_SDK_MODULES))
    def test_status_code_fallback_survives_missing_sdk(self, provider, block_sdk):
        block_sdk(*self._PROVIDER_SDK_MODULES[provider])
        # Even without the SDK, a status-carrying error still normalizes via the
        # SDK-free status-code fallback.
        result = normalize_provider_exception(fake_provider_error(429), provider)
        assert isinstance(result, RateLimitError)

    def test_normalize_context_manager_reraises_programmer_error_without_sdk(self, block_sdk):
        block_sdk("openai")
        original = TypeError("programmer bug")
        with pytest.raises(TypeError) as exc_info:
            with normalize_provider_errors("openai"):
                raise original
        assert exc_info.value is original


class TestExceptionMapOpenAI:
    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip("openai")

    def test_authentication_error(self):
        import openai
        _check_mapping(openai.AuthenticationError, "openai", AuthenticationError)

    def test_permission_denied(self):
        import openai
        _check_mapping(openai.PermissionDeniedError, "openai", PermissionDeniedError)

    def test_not_found(self):
        import openai
        _check_mapping(openai.NotFoundError, "openai", NotFoundError)

    def test_rate_limit(self):
        import openai
        _check_mapping(openai.RateLimitError, "openai", RateLimitError)

    def test_bad_request(self):
        import openai
        _check_mapping(openai.BadRequestError, "openai", BadRequestError)

    def test_internal_server_error(self):
        import openai
        _check_mapping(openai.InternalServerError, "openai", ServerError)

    def test_timeout(self):
        import openai
        _check_mapping(openai.APITimeoutError, "openai", APITimeoutError)

    def test_connection_error(self):
        import openai
        _check_mapping(openai.APIConnectionError, "openai", APIConnectionError)

    def test_response_validation(self):
        import openai
        _check_mapping(openai.APIResponseValidationError, "openai", ProviderResponseError)

    def test_length_finish_reason(self):
        import openai
        _check_mapping(openai.LengthFinishReasonError, "openai", GenerationError)

    def test_content_filter_finish_reason(self):
        import openai
        _check_mapping(openai.ContentFilterFinishReasonError, "openai", GenerationError)

    def test_conflict_error(self):
        import openai
        _check_mapping(openai.ConflictError, "openai", BadRequestError)

    def test_unprocessable_entity(self):
        import openai
        _check_mapping(openai.UnprocessableEntityError, "openai", BadRequestError)


class TestExceptionMapAnthropic:
    @pytest.fixture(autouse=True)
    def _require_anthropic(self):
        pytest.importorskip("anthropic")

    def test_authentication_error(self):
        import anthropic
        _check_mapping(anthropic.AuthenticationError, "anthropic", AuthenticationError)

    def test_permission_denied(self):
        import anthropic
        _check_mapping(anthropic.PermissionDeniedError, "anthropic", PermissionDeniedError)

    def test_not_found(self):
        import anthropic
        _check_mapping(anthropic.NotFoundError, "anthropic", NotFoundError)

    def test_rate_limit(self):
        import anthropic
        _check_mapping(anthropic.RateLimitError, "anthropic", RateLimitError)

    def test_bad_request(self):
        import anthropic
        _check_mapping(anthropic.BadRequestError, "anthropic", BadRequestError)

    def test_internal_server_error(self):
        import anthropic
        _check_mapping(anthropic.InternalServerError, "anthropic", ServerError)

    def test_overloaded(self):
        import anthropic._exceptions as anthropic_exc
        _check_mapping(anthropic_exc.OverloadedError, "anthropic", ServerError)

    def test_service_unavailable(self):
        import anthropic._exceptions as anthropic_exc
        _check_mapping(anthropic_exc.ServiceUnavailableError, "anthropic", ServerError)

    def test_deadline_exceeded_is_timeout(self):
        # DeadlineExceededError is a timeout, not a 5xx server error.
        import anthropic._exceptions as anthropic_exc
        _check_mapping(anthropic_exc.DeadlineExceededError, "anthropic", APITimeoutError)

    def test_timeout(self):
        import anthropic
        _check_mapping(anthropic.APITimeoutError, "anthropic", APITimeoutError)

    def test_connection_error(self):
        import anthropic
        _check_mapping(anthropic.APIConnectionError, "anthropic", APIConnectionError)

    def test_response_validation(self):
        import anthropic
        _check_mapping(anthropic.APIResponseValidationError, "anthropic", ProviderResponseError)

    def test_conflict_error(self):
        import anthropic
        _check_mapping(anthropic.ConflictError, "anthropic", BadRequestError)

    def test_request_too_large(self):
        import anthropic._exceptions as anthropic_exc
        _check_mapping(anthropic_exc.RequestTooLargeError, "anthropic", BadRequestError)

    def test_unprocessable_entity(self):
        import anthropic
        _check_mapping(anthropic.UnprocessableEntityError, "anthropic", BadRequestError)


class TestExceptionMapMistral:
    @pytest.fixture(autouse=True)
    def _require_mistral(self):
        pytest.importorskip("mistralai")

    def test_v2_errors_are_mapped(self, monkeypatch):
        # Remove this monkeypatch test when mistralai v2.x+ is the minimum.
        class HTTPValidationError(Exception):
            pass

        class ResponseValidationError(Exception):
            pass

        class NoResponseError(Exception):
            pass

        fake_errors = ModuleType("mistralai.client.errors")
        fake_errors.HTTPValidationError = HTTPValidationError
        fake_errors.ResponseValidationError = ResponseValidationError
        fake_errors.NoResponseError = NoResponseError
        monkeypatch.setitem(sys.modules, "mistralai.client.errors", fake_errors)
        _build_exception_map.cache_clear()

        assert isinstance(
            normalize_provider_exception(HTTPValidationError(), "mistral"),
            BadRequestError,
        )
        assert isinstance(
            normalize_provider_exception(ResponseValidationError(), "mistral"),
            ProviderResponseError,
        )
        assert isinstance(
            normalize_provider_exception(NoResponseError(), "mistral"),
            APIConnectionError,
        )

        _build_exception_map.cache_clear()

    def test_v2_non_exception_export_is_ignored(self, monkeypatch):
        # Remove this monkeypatch test when mistralai v2.x+ is the minimum.
        fake_errors = ModuleType("mistralai.client.errors")
        fake_errors.HTTPValidationError = object()
        monkeypatch.setitem(sys.modules, "mistralai.client.errors", fake_errors)
        _build_exception_map.cache_clear()

        assert not is_provider_exception(Exception("bad request"), "mistral")

        _build_exception_map.cache_clear()

    def test_httpx_timeout(self):
        import httpx
        _check_mapping(httpx.TimeoutException, "mistral", APITimeoutError)

    def test_httpx_connect_error(self):
        import httpx
        _check_mapping(httpx.ConnectError, "mistral", APIConnectionError)

    def test_status_code_fallback_401(self):
        # v1.9 compatibility is intentionally best-effort: HTTP-like errors
        # normalize through status-code fallback; v2 has explicit mappings above.
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


class TestExceptionMapGemini:
    @pytest.fixture(autouse=True)
    def _require_gemini(self):
        pytest.importorskip("google.genai")

    def test_server_error(self):
        from google.genai import errors as genai_errors
        _check_mapping(genai_errors.ServerError, "gemini", ServerError)

    def test_httpx_timeout(self):
        import httpx
        _check_mapping(httpx.TimeoutException, "gemini", APITimeoutError)

    def test_httpx_connect_error(self):
        import httpx
        _check_mapping(httpx.ConnectError, "gemini", APIConnectionError)

    def test_client_error_status_code_fallback(self):
        from google.genai import errors as genai_errors
        mock_exc = Mock(spec=genai_errors.ClientError)
        mock_exc.code = 401
        result = normalize_provider_exception(mock_exc, "gemini")
        assert isinstance(result, AuthenticationError)


class TestExceptionMapOllama:
    @pytest.fixture(autouse=True)
    def _require_ollama(self):
        pytest.importorskip("ollama")

    def test_builtin_connection_error(self):
        result = normalize_provider_exception(ConnectionError("unreachable"), "ollama")
        assert isinstance(result, APIConnectionError)

    def test_httpx_connect_error(self):
        import httpx
        result = normalize_provider_exception(httpx.ConnectError("unreachable"), "ollama")
        assert isinstance(result, APIConnectionError)

    def test_httpx_timeout(self):
        import httpx
        result = normalize_provider_exception(httpx.TimeoutException("timed out"), "ollama")
        assert isinstance(result, APITimeoutError)

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


class TestExceptionMapTGI:
    @pytest.fixture(autouse=True)
    def _require_tgi(self):
        pytest.importorskip("huggingface_hub")

    def test_inference_timeout(self):
        from huggingface_hub.errors import InferenceTimeoutError
        _check_mapping(InferenceTimeoutError, "tgi", APITimeoutError)

    def test_overloaded(self):
        from huggingface_hub.errors import OverloadedError
        _check_mapping(OverloadedError, "tgi", ServerError)

    def test_validation_error(self):
        from huggingface_hub.errors import ValidationError
        _check_mapping(ValidationError, "tgi", BadRequestError)

    def test_generation_error(self):
        from huggingface_hub.errors import GenerationError as HFGenerationError
        _check_mapping(HFGenerationError, "tgi", GenerationError)

    def test_incomplete_generation_error(self):
        from huggingface_hub.errors import IncompleteGenerationError
        _check_mapping(IncompleteGenerationError, "tgi", GenerationError)

    def test_bad_request_error(self):
        from huggingface_hub.errors import BadRequestError as HFBadRequestError
        _check_mapping(HFBadRequestError, "tgi", BadRequestError)

    def test_gated_repo_error(self):
        from huggingface_hub.errors import GatedRepoError
        _check_mapping(GatedRepoError, "tgi", PermissionDeniedError)

    def test_repository_not_found(self):
        from huggingface_hub.errors import RepositoryNotFoundError
        _check_mapping(RepositoryNotFoundError, "tgi", NotFoundError)

    def test_hf_hub_http_error_status_code_fallback(self):
        # HfHubHTTPError is the base class and is intentionally not in the
        # provider map; it falls through to the status-code fallback via
        # response.status_code.
        from huggingface_hub.errors import HfHubHTTPError
        mock_exc = Mock(spec=HfHubHTTPError)
        mock_exc.response = Mock(status_code=401)
        result = normalize_provider_exception(mock_exc, "tgi")
        assert isinstance(result, AuthenticationError)


class TestExceptionMapDottxt:
    @pytest.fixture(autouse=True)
    def _require_dottxt(self):
        pytest.importorskip("urllib3")

    def test_connect_timeout(self):
        from urllib3.exceptions import ConnectTimeoutError
        _check_mapping(ConnectTimeoutError, "dottxt", APITimeoutError)

    def test_read_timeout(self):
        from urllib3.exceptions import ReadTimeoutError
        _check_mapping(ReadTimeoutError, "dottxt", APITimeoutError)

    def test_max_retry_error(self):
        from urllib3.exceptions import MaxRetryError
        _check_mapping(MaxRetryError, "dottxt", APIConnectionError)

    def test_new_connection_error(self):
        from urllib3.exceptions import NewConnectionError
        _check_mapping(NewConnectionError, "dottxt", APIConnectionError)


class TestExceptionMapLMStudio:
    def test_connection_error(self):
        _check_mapping(ConnectionError, "lmstudio", APIConnectionError)

    def test_timeout_error(self):
        _check_mapping(TimeoutError, "lmstudio", APITimeoutError)

    def test_status_code_fallback(self):
        class LMStudioError(Exception):
            status_code = 429

        result = normalize_provider_exception(
            LMStudioError("rate limited"),
            "lmstudio",
        )
        assert isinstance(result, RateLimitError)


# vLLM and SGLang both reuse the OpenAI SDK client, so they share the OpenAI
# exception map. One parametrized class covers both providers.
@pytest.mark.parametrize("provider", ["vllm", "sglang"])
class TestExceptionMapOpenAICompatible:
    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip("openai")

    def test_authentication_error(self, provider):
        import openai
        _check_mapping(openai.AuthenticationError, provider, AuthenticationError)

    def test_rate_limit(self, provider):
        import openai
        _check_mapping(openai.RateLimitError, provider, RateLimitError)

    def test_provider_attribute_is_set(self, provider):
        import openai
        result = normalize_provider_exception(Mock(spec=openai.RateLimitError), provider)
        assert result.provider == provider


class TestLocalModelErrorPassthrough:
    """Local-runtime models intentionally do not normalize exceptions.

    Single guard test: ensures Transformers (the most-used local model)
    propagates raw exceptions unchanged. The same property holds for MLXLM
    and VLLMOffline by virtue of not importing ``normalize_provider_errors``;
    grep is the source of truth there.
    """

    def test_transformers_raises_raw(self):
        from outlines.models.transformers import Transformers

        class DummyModel:
            class config:
                is_encoder_decoder = False

            def generate(self, **kwargs):
                raise RuntimeError("local generation failed")

        model = Transformers.__new__(Transformers)
        model.model = DummyModel()

        with pytest.raises(RuntimeError, match="local generation failed"):
            model._generate_output_seq("", {"input_ids": object()})


class TestAPIErrorMessageConstruction:
    def test_explicit_message_used_as_is(self):
        err = APIError("custom message", provider="openai")
        assert str(err) == "custom message"

    def test_provider_and_original_exception(self):
        orig = ValueError("boom")
        err = APIError(provider="openai", original_exception=orig)
        assert "[openai]" in str(err)
        assert "boom" in str(err)

    def test_original_exception_only(self):
        orig = ValueError("boom")
        err = APIError(original_exception=orig)
        assert "boom" in str(err)
        assert "[" not in str(err)

    def test_request_id_extracted_from_original_exception(self):
        class E(Exception):
            request_id = "req-abc"
        err = APIError(original_exception=E())
        assert err.request_id == "req-abc"


class TestExtractRequestId:
    def test_request_id_attr(self):
        class E(Exception):
            request_id = "req-123"
        assert _extract_request_id(E()) == "req-123"

    def test_requestId_attr(self):
        class E(Exception):
            requestId = "req-456"
        assert _extract_request_id(E()) == "req-456"

    def test_no_request_id_returns_none(self):
        assert _extract_request_id(ValueError("no id")) is None

    def test_empty_string_request_id_ignored(self):
        class E(Exception):
            request_id = ""
        assert _extract_request_id(E()) is None

    def test_non_string_request_id_ignored(self):
        class E(Exception):
            request_id = 12345
        assert _extract_request_id(E()) is None

    def test_x_request_id_in_response_headers(self):
        class Resp:
            headers = {"x-request-id": "hdr-789"}
        class E(Exception):
            response = Resp()
        assert _extract_request_id(E()) == "hdr-789"

    def test_request_id_in_response_headers(self):
        class Resp:
            headers = {"request-id": "hdr-000"}
        class E(Exception):
            response = Resp()
        assert _extract_request_id(E()) == "hdr-000"

    def test_non_mapping_headers_ignored(self):
        class Resp:
            headers = "not-a-mapping"
        class E(Exception):
            response = Resp()
        assert _extract_request_id(E()) is None

    def test_mapping_headers_without_request_id_key(self):
        class Resp:
            headers = {"content-type": "application/json"}
        class E(Exception):
            response = Resp()
        assert _extract_request_id(E()) is None


class TestNormalizeProviderErrors:
    def test_provider_error_is_normalized(self):
        original = fake_provider_error(429)

        with pytest.raises(RateLimitError) as exc_info:
            with normalize_provider_errors("openai"):
                raise original

        assert exc_info.value.original_exception is original

    def test_non_provider_error_passes_through(self):
        original = TypeError("programmer error")

        with pytest.raises(TypeError) as exc_info:
            with normalize_provider_errors("openai"):
                raise original

        assert exc_info.value is original

    def test_generator_error_is_normalized(self):
        original = fake_provider_error(429)

        def stream():
            with normalize_provider_errors("openai"):
                raise original
                yield  # pragma: no cover

        with pytest.raises(RateLimitError) as exc_info:
            next(stream())

        assert exc_info.value.original_exception is original

    def test_generator_success_yields_through_context_manager(self):
        events = []

        def stream():
            with normalize_provider_errors("openai"):
                events.append("entered")
                yield "one"
                events.append("after-one")
                yield "two"
                events.append("after-two")
            events.append("exited")

        assert list(stream()) == ["one", "two"]
        assert events == ["entered", "after-one", "after-two", "exited"]

    @pytest.mark.asyncio
    async def test_async_error_is_normalized(self):
        original = fake_provider_error(429)

        async def generate():
            with normalize_provider_errors("openai"):
                raise original

        with pytest.raises(RateLimitError) as exc_info:
            await generate()

        assert exc_info.value.original_exception is original

    @pytest.mark.asyncio
    async def test_async_generator_success_yields_through_context_manager(self):
        events = []

        async def stream():
            with normalize_provider_errors("openai"):
                events.append("entered")
                yield "one"
                events.append("after-one")
                yield "two"
                events.append("after-two")
            events.append("exited")

        assert [chunk async for chunk in stream()] == ["one", "two"]
        assert events == ["entered", "after-one", "after-two", "exited"]
