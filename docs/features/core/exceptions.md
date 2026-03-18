---
title: Error Handling
---

# Error Handling

Native runtime exceptions are preserved for the following in-process local
runtimes:

  * `LlamaCpp`
  * `MLXLM`
  * `transformers`
  * `LMStudio`

But for the remaining **API-backed integrations** (Mistral, OpenAI, Ollama,
etc), Outlines provides a common exception hierarchy under `OutlinesError`, as
follows.

## Exception hierarchy

All classes live in `outlines.exceptions`.

```
OutlinesError
└── APIError
    ├── AuthenticationError
    ├── PermissionDeniedError
    ├── NotFoundError
    ├── RateLimitError
    ├── BadRequestError
    ├── ServerError
    ├── APITimeoutError
    ├── APIConnectionError
    ├── ProviderResponseError
    └── GenerationError
```

## Reference

| Class                   | Category            | HTTP status(es)               | Suggested user action                                                                       |
| ----------------------- | ------------------- | ----------------------------- | ------------------------------------------------------------------------------------------- |
| `AuthenticationError`   | Client error        | 401                           | Fix input: Check API key.                                                                    |
| `PermissionDeniedError` | Client error        | 403                           | Fix input: Check permissions for API key.                                                    |
| `NotFoundError`         | Client error        | 404                           | Fix input: Confirm model name, endpoint, etc.                                                |
| `RateLimitError`        | Client error        | 429                           | Retry: Slow down and retry, or reduce request frequency, batch size, etc.                   |
| `BadRequestError`       | Client error        | 400, 409, 413, 422, other 4xx | Fix input: Check prompt length, schema, unsupported parameters, etc.                        |
| `ServerError`           | Server error        | 5xx, 529 (Anthropic overload) | Retry: Perhaps retry after a short wait.                                                     |
| `APITimeoutError`       | Network/transport   | — (no HTTP status)            | Retry: Provider may be overloaded.                                                           |
| `APIConnectionError`    | Network/transport   | — (no HTTP status)            | Retry: Could not reach provider, check connection.                                           |
| `ProviderResponseError` | Response/generation | — (no HTTP status)            | Provider bug: May be a temporary issue or schema/format mismatch.                           |
| `GenerationError`       | Response/generation | — (no HTTP status)            | Inspect output: Output likely hit a content filter, model's max-token limit, or similar.    |

## Usage

Normalized `APIError` subclasses are raised by Outlines model wrappers; if you call a provider SDK directly, you should expect provider-native exceptions instead.

```python
from outlines.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ServerError,
    APITimeoutError,
    APIConnectionError,
)

try:
    result = model("Hello")
except AuthenticationError:
    print("Check your API key")
except RateLimitError as e:
    print(f"Rate limited by {e.provider}, back off and retry")
except APIError as e:
    if e.retryable:
        print(f"Transient error from {e.provider} (status={e.status_code}), retry")
    else:
        raise
```

## `retryable` flag

Four exception types set `retryable = True`: `RateLimitError`, `ServerError`, `APITimeoutError`, and `APIConnectionError`. You can branch on this attribute without hardcoding a list of types:

```python
except APIError as e:
    if e.retryable:
        time.sleep(backoff)
        retry()
    else:
        raise
```

## Available attributes

Every `APIError` instance exposes:

| Attribute            | Type            | Description                                   |
| -------------------- | --------------- | --------------------------------------------- |
| `provider`           | `str \| None`   | Provider name, e.g. `"mistral"`, `"openai"` |
| `original_exception` | `Exception \| None` | The raw SDK exception                     |
| `status_code`        | `int \| None`   | HTTP status code, if available                |
| `request_id`         | `str \| None`   | Provider request ID, if available             |
| `retryable`          | `bool`          | Whether the error is worth retrying           |
| `hint`               | `str`           | Actionable suggestion shown in tracebacks                                 |
