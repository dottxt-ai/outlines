# Mistral AI Integration

Outlines now supports models from [Mistral AI](https://mistral.ai) via the official `mistralai` package. This integration provides a thin wrapper around the `mistralai.Mistral` client, handling prompt formatting, structured outputs, and streaming.

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

# Initialize the client and model
client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
model = from_mistral(client, "mistral-small-latest")

# Generate text
result = model.generate("What is the capital of France?")
print(result)
```

## Available Models

As of September 2025, Mistral AI offers a range of open and commercial models. Common ones include:

- **Open models (free for commercial use):**
  - `open-mistral-7b`
  - `open-mixtral-8x7b`
  - `open-mixtral-8x22b`
  - `codestral-latest` (⚠ does not support structured outputs, AVOID for JSON schemas)

- **Premier/commercial models (API access required):**
  - `mistral-tiny`
  - `mistral-small-latest` (efficient general-purpose)
  - `mistral-large-latest` (frontier-class for complex tasks)
  - `pixtral-12b-2409` (multimodal with image understanding)

For the full, up-to-date list, refer to the [Mistral AI Models Overview](https://docs.mistral.ai/getting-started/models/models_overview/).

## Configuration

Create the model with optional model name:

```python
model = from_mistral(
    client,
    "mistral-large-latest"
)
```

**Note:** Inference parameters (like `temperature`, `max_tokens`) are passed directly to `generate()` calls, not during model initialization.

## Structured Generation

Mistral supports structured JSON outputs via JSON schemas. Outlines automatically converts Pydantic models, dataclasses, TypedDicts, Genson schemas, and JSON schemas into the required format.

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
    output_type=Person
)
```

**Limitations:**
- **Regex and CFG-based structured outputs are not supported** by the Mistral API. Use open-source models or other integrations instead.
- Structured outputs are **unavailable** for `codestral-*` models (they lack JSON schema support).
- For best results with complex/nested schemas, use `mistral-large-latest` or higher; simpler models may require retries.

Check model compatibility:
```python
if model.supports_structured_output():
    # Safe to use Pydantic/JSON schemas
    pass
else:
    print("Use text-only generation")
```

## Streaming

Stream responses chunk by chunk:

```python
for chunk in model.generate_stream("Write a short story about AI"):
    print(chunk, end="", flush=True)
```

Streaming works with both text and structured outputs (JSON chunks are yielded as text; parse manually if needed).

## Batch Processing

Batch inference is currently **not supported** by the `mistralai` library:

```python
# This will raise NotImplementedError
# model.generate_batch(prompts)

# Use sequential processing instead
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
    print(f"Input Error: {e}")  # e.g., invalid message roles or content, empty lists
```

For schema errors (e.g., "Mistral does not support your schema"), simplify your Pydantic model or switch models.

## Advanced Usage

### Custom Inference Parameters

Pass parameters directly to `generate`:

```python
result = model.generate(
    "Write a poem",
    temperature=0.9,  # More creative
    max_tokens=200,
    top_p=0.95
)
```

### Multimodal Inputs (Images)

Mistral supports image inputs via `content` lists (e.g., for `pixtral-*` models). Outlines handles this in `Chat` or list inputs:

```python
from outlines.inputs import Image
from PIL import Image as PILImage

# Load image
pil_image = PILImage.open("path/to/image.jpg")
image = Image(pil_image)

# Text + image prompt
result = model.generate(
    ["Describe this image", image],
    model_name="pixtral-12b-2409"
)

# Or via Chat
chat = Chat([
    {"role": "user", "content": ["What do you see?", image]}
])
result = model.generate(chat)
```

**Note:** Image support requires compatible models like `pixtral-12b-2409`. Text-only models will raise errors. The first item must be a string, followed by optional `Image` objects.

### Chat Messages

Use `Chat` objects for multi-turn conversations:

```python
from outlines.inputs import Chat

chat = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "Tell me more about its uses."}
])

response = model.generate(chat)
```

**Supported roles:** `system`, `user`, `assistant`. Other roles raise `ValueError`.

### Async Support

For asynchronous usage:

```python
from outlines.models.mistral import from_mistral

async_model = from_mistral(client, "mistral-small-latest", async_client=True)

result = await async_model.generate("Hello, world!")
```

## Input Types

The integration supports three input formats:

- **String**: Simple prompts `"Hello, world!"`
- **List**: Text + images `["Describe this", Image(pil_image)]`
- **Chat**: Multi-turn conversations with roles

**Validation:**
- List inputs must start with a string, followed by optional `Image` objects
- Empty lists raise `ValueError: "Content list cannot be empty."`
- Non-string first items raise `ValueError: "The first item in the list should be a string."`

## Troubleshooting

- **ImportError: No module named 'mistralai'**
  → Run `pip install mistralai`.

- **Authentication Error**
  → Verify `MISTRAL_API_KEY` is set and valid. Test with the [Mistral Playground](https://chat.mistral.ai).

- **Schema Error (e.g., "does not support your schema")**
  → Avoid regex/CFG; ensure no `pattern` fields in Pydantic (Outlines sets `additionalProperties: false`); try a simpler schema or different model.

- **Model Not Found Error**
  → Confirm the model name (e.g., `"mistral-small-latest"`) and your subscription tier. Check [docs](https://docs.mistral.ai/getting-started/models/).

- **Rate Limits or Quotas**
  → Monitor usage in the Mistral console; upgrade your plan for higher limits.

- **Input Validation Errors**
  → Ensure Chat messages use valid roles (`system`, `user`, `assistant`); list inputs start with strings.

## Notes

- Outlines automatically sets `additionalProperties: false` in JSON schemas for stricter parsing.
- Supports `str`, `list` (text+images), and `Chat` inputs; converts to Mistral message objects.
- For multimodal inputs, use compatible models like `pixtral-12b-2409` with `Image` objects.
- `generate_batch` raises `NotImplementedError` due to API limitations.
- **Author:** Steven E. Elliott aka see (seeyallc6c@gmail.com)

---

For integration examples, see the [earnings-reports demo](https://github.com/dottxt-ai/demos/tree/main/earnings-reports) (requires modification for Mistral).

*Last updated: September 19, 2025*
