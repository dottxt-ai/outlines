# OpenAI-Compatible APIs

Many inference providers offer OpenAI-compatible APIs, allowing you to use the familiar OpenAI SDK while connecting to different backends. Outlines allows you can leverage various providers while maintaining consistent code.

## What are OpenAI-Compatible APIs?

OpenAI-compatible APIs implement the same REST endpoints and request/response formats as OpenAI's API, but serve different models or run on different infrastructure. This allows you to use the `openai` Python library with any compatible provider by simply changing the `base_url`.

!!! Installation

    You need to install the `openai` library to be able to use the OpenAI-compatible APIs in Outlines. Install all optional dependencies of the `OpenAI` model with: `pip install outlines[openai]`.

## General Usage Pattern

The standard approach is to use the OpenAI SDK with a custom base URL:

```python
import openai
import outlines

# Point to your OpenAI-compatible endpoint
client = openai.OpenAI(
    base_url="https://your-provider.com/v1",  # Custom endpoint
    api_key="your-api-key"
)

# Use with Outlines
model = outlines.from_openai(client, "model-name")
```

## Important: Provider-Specific Parameters

!!! Warning "API-Specific Parameters"

    Some providers require additional parameters in the API request for structured generation to work properly. These are typically passed as extra arguments when calling the model.

    For example, some providers may need special parameters in the request body to enable guided generation or specify constraints. Always consult your provider's documentation for structured generation requirements.

## Popular OpenAI-Compatible Providers

Many providers offer OpenAI-compatible endpoints:

- **Groq**
- **Together AI**
- **Anyscale**
- **Fireworks AI**
- **Perplexity**
- **Local servers** (LocalAI, etc.)

## Configuration Examples

### Basic Setup
```python
import openai
import outlines

# Generic OpenAI-compatible setup
client = openai.OpenAI(
    base_url="https://api.your-provider.com/v1",
    api_key="your-api-key"
)

model = outlines.from_openai(client, "provider-model-name")
```

### With Authentication Headers
```python
import openai
import outlines

# Some providers need custom headers
client = openai.OpenAI(
    base_url="https://api.your-provider.com/v1",
    api_key="your-api-key",
    default_headers={"Custom-Header": "value"}
)

model = outlines.from_openai(client, "provider-model-name")
```

## Related Documentation

For specific implementations that use OpenAI-compatible APIs:

- [SGLang](sglang.md): Local inference server with OpenAI-compatible endpoints
- [vLLM](vllm.md): High-performance inference with OpenAI-compatible API
- [OpenAI](openai.md): The original OpenAI API implementation
