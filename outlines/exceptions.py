"""Outlines exception hierarchy and per-provider normalization.

All public exceptions inherit from ``APIError`` → ``OutlinesError`` → ``Exception``.
Use :func:`normalize_provider_exception` to convert a raw provider SDK exception
into the appropriate Outlines type.
"""

from collections.abc import Mapping

__all__ = [
    "OutlinesError",
    "APIError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "RateLimitError",
    "BadRequestError",
    "ServerError",
    "APITimeoutError",
    "APIConnectionError",
    "ProviderResponseError",
    "GenerationError",
    "normalize_provider_exception",
]


class OutlinesError(Exception):
    pass


class APIError(OutlinesError):
    """Base class for all provider API errors raised by Outlines.

    Subclasses map to specific HTTP status codes or failure categories
    (see the hierarchy below). Catch this class to handle any provider
    error generically, or catch a subclass for finer-grained control.

    Attributes
    ----------
    provider : str | None
        Short provider name, e.g. ``"openai"`` or ``"anthropic"``.
    original_exception : Exception | None
        The raw SDK exception that was caught, preserved for debugging.
    status_code : int | None
        HTTP status code, if one could be extracted from the exception.
    request_id : str | None
        Provider request ID extracted from the exception or response headers,
        useful when filing bug reports with a provider.
    retryable : bool
        ``True`` for transient errors worth retrying (rate limits, timeouts,
        5xx server errors, connection failures). ``False`` for permanent errors
        that require fixing the request or credentials.
    hint : str
        Short human-readable suggestion. On Python 3.11+ this is attached via
        ``add_note()`` and displayed as ``  → <hint>`` on its own line in
        tracebacks; on 3.10 it is stored as ``_hint_note``.
    """

    retryable: bool = False  # overridden to True on transient error subclasses
    hint: str = ""           # overridden on each subclass with actionable advice

    def __init__(
        self,
        message: str | None = None,
        provider: str | None = None,
        original_exception: Exception | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
    ):
        if message is None:
            if provider and original_exception is not None:
                message = f"API Error [{provider}]: {original_exception}"
            elif provider:
                message = f"API Error [{provider}]"
            elif original_exception is not None:
                message = f"API error: {original_exception}"
            else:
                message = "API error"

        super().__init__(message)

        # PEP 678 (Python 3.11+): notes appear on their own line in tracebacks
        if self.hint:
            note = f"  → {self.hint}"
            if hasattr(self, "add_note"):  # Python 3.11+
                self.add_note(note)
            else:
                self._hint_note = note

        self.provider = provider
        self.original_exception = original_exception
        self.status_code = status_code or _extract_status_code(original_exception)
        self.request_id = request_id

        if original_exception is not None:
            self.request_id = self.request_id or _extract_request_id(original_exception)


# --- Client errors (4xx) ---

class AuthenticationError(APIError):
    """401 - bad or missing API key."""
    hint = "Check API key."


class PermissionDeniedError(APIError):
    """403 - valid key, insufficient scope."""
    hint = "Check permissions for API key."


class NotFoundError(APIError):
    """404 - wrong model name or endpoint."""
    hint = "Confirm model name, endpoint, etc."


class RateLimitError(APIError):
    """429 - rate limit exceeded."""
    retryable = True
    hint = "Slow down and retry, or reduce request frequency, batch size, etc."


class BadRequestError(APIError):
    """400, 409, 413, 422, other 4xx - malformed request."""
    hint = "Check prompt length, schema, unsupported parameters, etc."


# --- Server errors (5xx) ---

class ServerError(APIError):
    """500-599, Anthropic 529 - server-side failure."""
    retryable = True
    hint = "Perhaps retry after a short wait."


# --- Network / transport ---

class APITimeoutError(APIError):
    """Request or connect timeout."""
    retryable = True
    hint = "Provider may be overloaded."


class APIConnectionError(APIError):
    """Unreachable host, DNS failure, or refused connection."""
    retryable = True
    hint = "Could not reach provider, check connection."


# --- Response / generation ---

class ProviderResponseError(APIError):
    """Malformed or unparseable response from the provider."""
    hint = "May be a temporary issue or schema/format mismatch."


class GenerationError(APIError):
    """Content filter hit or length stop."""
    hint = "Output likely hit a content filter, model's max-token limit, or similar."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_status_code(exc: Exception | None) -> int | None:
    if exc is None:
        return None
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    return None


def _extract_request_id(exc: Exception) -> str | None:
    for attr in ("request_id", "requestId", "x_request_id", "x-request-id"):
        if isinstance(value := getattr(exc, attr, None), str) and value:
            return value

    headers = getattr(getattr(exc, "response", None), "headers", None)
    if isinstance(headers, Mapping):
        normalized_headers = {
            str(key).lower().replace("_", "-"): value
            for key, value in headers.items()
        }
        for key in ("x-request-id", "request-id"):
            if isinstance(value := normalized_headers.get(key), str) and value:
                return value

    return None


# ---------------------------------------------------------------------------
# Status-code fallback map (used by normalize_provider_exception)
# ---------------------------------------------------------------------------

_STATUS_CODE_MAP: dict[int, type[APIError]] = {
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
    # Range-based fallbacks (4xx → BadRequestError, 5xx → ServerError) are
    # handled by the if-blocks in normalize_provider_exception below.
}


# ---------------------------------------------------------------------------
# Per-provider SDK exception maps
# ---------------------------------------------------------------------------

def _build_exception_map(provider: str) -> dict[type, type[APIError]]:
    """Return the SDK-exception → Outlines-exception mapping for *provider*.

    Imports are lazy so no SDK is loaded until an exception actually occurs.
    vLLM and SGLang reuse the OpenAI map because they use the OpenAI SDK client.
    """
    if provider in ("openai", "vllm", "sglang"):  # vLLM and SGLang use the OpenAI SDK client
        import openai
        return {
            openai.AuthenticationError: AuthenticationError,
            openai.PermissionDeniedError: PermissionDeniedError,
            openai.NotFoundError: NotFoundError,
            openai.RateLimitError: RateLimitError,
            openai.BadRequestError: BadRequestError,
            openai.ConflictError: BadRequestError,
            openai.UnprocessableEntityError: BadRequestError,
            openai.InternalServerError: ServerError,
            openai.APITimeoutError: APITimeoutError,
            openai.APIConnectionError: APIConnectionError,
            openai.APIResponseValidationError: ProviderResponseError,
            openai.LengthFinishReasonError: GenerationError,
            openai.ContentFilterFinishReasonError: GenerationError,
        }

    if provider == "anthropic":
        import anthropic
        import anthropic._exceptions as anthropic_exc
        return {
            anthropic.AuthenticationError: AuthenticationError,
            anthropic.PermissionDeniedError: PermissionDeniedError,
            anthropic.NotFoundError: NotFoundError,
            anthropic.RateLimitError: RateLimitError,
            anthropic.BadRequestError: BadRequestError,
            anthropic.ConflictError: BadRequestError,
            anthropic_exc.RequestTooLargeError: BadRequestError,
            anthropic.UnprocessableEntityError: BadRequestError,
            anthropic.InternalServerError: ServerError,
            anthropic_exc.ServiceUnavailableError: ServerError,
            anthropic_exc.DeadlineExceededError: ServerError,
            anthropic_exc.OverloadedError: ServerError,
            anthropic.APITimeoutError: APITimeoutError,
            anthropic.APIConnectionError: APIConnectionError,
            anthropic.APIResponseValidationError: ProviderResponseError,
        }

    if provider == "mistral":
        import httpx
        import mistralai.models as mm
        return {
            mm.HTTPValidationError: BadRequestError,
            mm.ValidationError: BadRequestError,
            mm.ResponseValidationError: ProviderResponseError,
            mm.NoResponseError: APIConnectionError,
            httpx.TimeoutException: APITimeoutError,
            httpx.ConnectError: APIConnectionError,
            # MistralError / SDKError with .status_code handled by status-code fallback
        }

    if provider == "gemini":
        import httpx
        import aiohttp
        from google.genai import errors as genai_errors
        return {
            genai_errors.ServerError: ServerError,
            # genai_errors.ClientError carries .code with the HTTP status; it is not
            # listed here so it falls through to the status-code fallback in
            # normalize_provider_exception, which maps it to the right subclass.
            httpx.TimeoutException: APITimeoutError,
            httpx.ConnectError: APIConnectionError,
            aiohttp.ServerTimeoutError: APITimeoutError,
            aiohttp.ClientConnectorError: APIConnectionError,
        }

    if provider == "ollama":
        import ollama
        return {
            ollama.RequestError: APIConnectionError,
            # ollama.ResponseError.status_code handled by status-code fallback
            # (.status_code defaults to -1 when unknown; range guard filters it out)
        }

    if provider == "tgi":
        import huggingface_hub.errors as hf_errors
        return {
            hf_errors.InferenceTimeoutError: APITimeoutError,
            hf_errors.OverloadedError: ServerError,
            hf_errors.ValidationError: BadRequestError,
            hf_errors.GenerationError: GenerationError,
            hf_errors.IncompleteGenerationError: GenerationError,
            hf_errors.BadRequestError: BadRequestError,
            hf_errors.GatedRepoError: PermissionDeniedError,
            hf_errors.RepositoryNotFoundError: NotFoundError,
            # hf_errors.HfHubHTTPError (base) carries .response.status_code; handled by fallback
        }

    if provider == "dottxt":
        import urllib3.exceptions as urllib3_exc
        return {
            # NewConnectionError subclasses ConnectTimeoutError, so it must come first
            urllib3_exc.NewConnectionError: APIConnectionError,
            urllib3_exc.ConnectTimeoutError: APITimeoutError,
            urllib3_exc.ReadTimeoutError: APITimeoutError,
            urllib3_exc.MaxRetryError: APIConnectionError,
        }

    # Unknown provider: no SDK-specific mapping available.
    # normalize_provider_exception will fall back to status-code inspection.
    return {}


def normalize_provider_exception(exc: Exception, provider: str) -> APIError:
    """Map a provider SDK exception to the appropriate Outlines exception.

    1. Try the provider's SDK exception map (most-specific-first via isinstance).
    2. Fall back to status-code inspection on the original exception.
    3. Default to generic APIError.
    """
    for provider_exc_cls, outlines_exc_cls in _build_exception_map(provider).items():
        if isinstance(exc, provider_exc_cls):
            return outlines_exc_cls(provider=provider, original_exception=exc)

    code = _extract_status_code(exc)
    if code is not None:
        if code in _STATUS_CODE_MAP:
            return _STATUS_CODE_MAP[code](provider=provider, original_exception=exc)
        if code >= 500:
            return ServerError(provider=provider, original_exception=exc)
        if 400 <= code < 500:
            return BadRequestError(provider=provider, original_exception=exc)

    return APIError(provider=provider, original_exception=exc)
