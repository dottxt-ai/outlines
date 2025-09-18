# Mistral AI Integration

Outlines now supports models from [Mistral AI](https://mistral.ai) via the official `mistralai` package.
This integration provides a thin wrapper around the `mistralai.Mistral` client, handling prompt formatting, structured outputs, and streaming.

## Installation

```bash
pip install mistralai
```

## Basic Usage

```python
import os
from mistralai import Mistral as MistralClient
from outlines.models.mistral import from_mistral

# Set your API key
os.environ["MISTRAL_API_KEY"] = "your-api-key-here"
# or in terminal: export MISTRAL_API_KEY=YourRawKeyHere

# Initialize the client and model
client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
model = from_mistral(client, "mistral-large-latest")

# Generate text
result = model.generate("What is the capital of France?")
print(result)
```

## Available Models

As of September 2025, Mistral AI offers a range of open and commercial models. Common ones include:

- **Open models (free for commercial use):**
  - open-mistral-7b
  - open-mixtral-8x7b
  - open-mixtral-8x22b
  - codestral-latest (⚠ does not support structured outputs, AVOID for JSON schemas)
  - devstral-small (SOTA coding model, released May 2025)

- **Premier/commercial models (API access required):**
  - mistral-tiny
  - mistral-small-latest (or mistral-small-3.1, efficient general-purpose)
  - mistral-medium-latest (or mistral-medium-3, released May 2025; excels in coding/STEM, multimodal)
  - mistral-large-latest (or mistral-large-2, frontier-class for complex tasks)
  - pixtral-large (multimodal with image understanding, released 2024)
  - magistral (reasoning models, launched June 2025)
  - voxtral-small (audio/chat model, released July 2025)
  - mistral-saba (optimized for Middle East/South Asia languages, released February 2025)

For the full, up-to-date list, refer to the [Mistral AI Models Overview](https://docs.mistral.ai/getting-started/models/models_overview/). Note that models may be deprecated or retired over time (e.g., older versions like mistral-medium are phased out).

## Configuration

You can set default inference parameters when creating the model:

```python
model = from_mistral(
    client,
    "mistral-large-latest"
)
```

## Structured Generation

Mistral supports structured JSON outputs via JSON schemas and Pydantic parsing. Outlines automatically converts Pydantic models, dataclasses, TypedDicts, and JSON schemas into the required format. Use `chat.complete` for JSON objects or `chat.parse` for direct Pydantic validation.

```python
from typing import Literal
from pydantic import BaseModel

# Multiple choice (via enum in JSON schema)
sentiment = model.generate(
    "The movie was absolutely fantastic!",
    output_type=Literal["positive", "negative", "neutral"]
)


# Structured object (Pydantic model)
class Person(BaseModel):
    name: str
    age: int
    occupation: str

person = model.generate(
    "Generate information about a software engineer",
    Person
)
```

⚠️ **Limitations:**
- Regex and CFG-based structured outputs are not supported by the Mistral API. Use open-source models or other integrations instead.
- Structured outputs are unavailable for `codestral-*` models (they lack JSON schema support).
- For best results with complex/nested schemas, use `mistral-large-latest` or higher; simpler models may require retries.

## Streaming

Stream responses chunk by chunk:

```python
for chunk in model.generate_stream("Write a short story about AI"):
    print(chunk, end="", flush=True)
```

Streaming works with both text and structured outputs (JSON chunks are yielded as text; parse manually if needed).

## Batch Processing

Batch inference is currently not supported by the `mistralai` library.
Process prompts sequentially or use multiple `generate` calls for lists:

```python
results = [model.generate(prompt) for prompt in prompts]
```

## Error Handling

The integration raises specific exceptions for common issues:

```python
try:
    result = model.generate("Hello", output_type=Person)
except RuntimeError as e:
    print(f"API Error: {e}")  # e.g., rate limits, network issues
except TypeError as e:
    print(f"Schema/Output Error: {e}")  # e.g., unsupported schema, model incompatibility
except ValueError as e:
    print(f"Input Error: {e}")  # e.g., invalid message roles or content
```

For schema errors (e.g., "Invalid schema format"), simplify your Pydantic model or switch models.

## Advanced Usage

### Custom System Prompts

Prepend a system message to all inputs:

```python
model = from_mistral(
    client,
    "mistral-large-latest",
    system_prompt="You are an expert Python programmer."
)

code = model.generate("Write a function to calculate fibonacci numbers")
```

If your `Chat` input already includes a system message, the adapter's prompt is skipped to avoid duplication.

### Multimodal Inputs (Images)

Mistral supports image inputs via `content` lists in messages (e.g., for `pixtral-*` models). Outlines handles this in `Chat` or list inputs:

```python
from outlines.inputs import Image

chat = Chat([
    {"role": "user", "content": ["Describe this image.", Image("path/to/image.jpg")]}
])
result = model.generate(chat, model_name="pixtral-large")
```

Note: Image support requires compatible models like `pixtral-large`; text-only models will raise errors.

### Temperature and Randomness Control

```python
# Deterministic output
det_model = from_mistral(
    client,
    "mistral-large-latest",
    config={"temperature": 0.1, "top_p": 0.1}
)

# Creative generation
creative_model = from_mistral(
    client,
    "mistral-large-latest",
    config={"temperature": 0.9, "top_p": 0.95}
)
```

Inference kwargs (e.g., `temperature=0.5`) override config defaults in `generate`.

### Checking Structured Output Support

```python
if model.supports_structured_output("codestral-latest"):
    # Safe to use Pydantic/JSON
    pass
else:
    print("Use text-only generation")
```

## Troubleshooting

- **ImportError: No module named 'mistralai'**
  → Run `pip install mistralai`.

- **Authentication Error**
  → Verify `MISTRAL_API_KEY` is set and valid. Test with the Mistral docs.

- **Schema Error (e.g., "does not support your schema")**
  → Avoid regex/CFG; ensure no `additionalProperties: true` (Outlines sets it to `false`); try a simpler schema or different model.

- **Model Not Found Error**
  → Confirm the model name (e.g., "mistral-large-latest") and your subscription tier. Check [docs](https://docs.mistral.ai/getting-started/models/).

- **Rate Limits or Quotas**
  → Monitor usage; upgrade your plan for higher limits.

- **Deprecation Warnings**
  → Mistral retires older models periodically. Update to `-latest` variants.

For integration examples, a working modification to the dottxt-ai/demos/earnings-reports demo can be obtained by contacting the author.

## Notes

- Outlines automatically sets `additionalProperties: false` in JSON schemas for stricter parsing.
- Supports `Chat`, string, and list inputs; converts to Mistral message objects.
- For audio/multimodal (e.g., `voxtral-*`, `pixtral-*`), use compatible models and content lists.
- `generate_batch` raises `NotImplementedError` due to API limitations.
- Author: Steven E. Elliott aka see (seeyallc6c@gmail.com)

---
