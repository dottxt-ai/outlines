// # Mistral AI Models

Outlines now supports models from [Mistral AI](https://mistral.ai) via the official `mistralai` package.
This implementation provides a thin wrapper around the `mistralai.Mistral` client, handling prompt formatting, structured outputs, and streaming.
This document describes how to use the **Mistral SDK** with Outlines, both in
synchronous and asynchronous contexts.

---

## 0. Installation

```bash
pip install mistralai
export MISTRAL_API_KEY=YOUR_API_KEY_GOES_HERE # Or in code if done securely
```


---

## 1. Open Models

Below is a list of the open-weight models available via Apache2 license released by Mistral. These can be
used directly via the API or downloaded where applicable (as of 2025-09-19).

| API Endpoint             | Version | Model                | Max Tokens | Description                                                                | Outlines Support   | Multi-Modal        |
|--------------------------|---------|----------------------|------------|----------------------------------------------------------------------------|--------------------|--------------------|
| `magistral-small-2509`   | 25.09   | Magistral Small 1.2  | 128k       | Small reasoning model (Sept 2025) with vision support.                     | ✔️                 | ✔️                 |
| `magistral-small-2507`   | 25.07   | Magistral Small 1.1  | 40k        | Small reasoning model (July 2025).                                         | ✔️                 | ❌                 |
| `voxtral-mini-2507`      | 25.07   | Voxtral Mini         | 32k        | Mini audio input model.                                                    | ❓                 | ❓                 |
| `mistral-small-2506`     | 25.06   | Mistral Small 3.2    | 128k       | Updated small model (June 2025).                                           | ✔️                 | ✔️                 |
| `magistral-small-2506`   | 25.06   | Magistral Small 1    | 40k        | First small reasoning model (June 2025).                                   | ✔️                 | ✔️                 |
| `devstral-small-2507`    | 25.07   | Devstral Small 1.1   | 128k       | Tool-use model for code exploration and editing (July 2025).               | ❓                 | ❓                 |
| `mistral-small-2503`     | 25.03   | Mistral Small 4    | 128k       | Small model with image understanding (March 2025).                         | ✔️                 | ✔️                 |
| `mistral-small-2501`     | 25.01   | Mistral Small 3      | 32k        | Small model (Jan 2025).                                                    | ✔️                 | ❌                 |
| `devstral-small-2505`    | 25.05   | Devstral Small 1     | 128k       | Open source 24B text model for tool-use and software agents (June 2025).   | ❓                 | ❓                 |
| `pixtral-12b-2409`       | 24.09   | Pixtral 12B          | 128k       | 12B multimodal model with image understanding (Sept 2024).                 | ✔️                 | ✔️                 |
| `open-mistral-nemo`      | 24.07   | Mistral Nemo 12B     | 128k       | Multilingual open source model (July 2024).                                | ✔️                 | ❓                 |


**Mistral SDK Limitations:**
- Regex and CFG-based structured outputs are not supported by the Mistral API.
- Structured outputs (including Pydantic/dataclass-based ones) are unavailable for codestral-mamba models, because they lack JSON Schema support.

**Outlines Expansion:**
- Outlines provides a drop-in for Regex and CFG-based structured outputs, reducing the SDK limitations.

---
## 2. Listing Available Models via SDK

The mistralai python SDK provides methods to query the API for a list of [all available models](https://docs.mistral.ai/getting-started/models/models_overview/#api-versioning),
including **paid endpoints** for [premium models](https://docs.mistral.ai/getting-started/models/models_overview/) in addition to open weights.

##### Sync Example
```python
#!/usr/bin/env python
import os
from mistralai import Mistral

def main():
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    list_models_response = client.models.list()
    print(list_models_response)

if __name__ == "__main__":
    main()
```

##### Async Example
```python
#!/usr/bin/env python
import asyncio
import os
from mistralai import Mistral

async def main():
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    list_models_response = await client.models.list_async()
    print(list_models_response)

if __name__ == "__main__":
    asyncio.run(main())
```
---
## 3. Basic Usage


### 3.1 mistralai SDK wrapper

Generate outputs for the given model name :

#### Sync
```python
from mistralai import Mistral
from outlines.models.mistral import from_mistral

def main():
    client = Mistral(api_key="YOUR_API_KEY")
    model = from_mistral(client, "mistral-small-2506")
    result = model.generate("What is 2+2? Answer in one sentence.")
    print(result)

if __name__ == "__main__":
    main()
```

#### Async
```python
import asyncio
from mistralai import Mistral
from outlines.models.mistral import from_mistral

async def main():
    client = Mistral(api_key="YOUR_API_KEY")
    model = from_mistral(client, "mistral-small-2506", async_client=True)
    result = await model.generate("What is 2+2? Answer in one sentence.")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```


### 3.2 Structured Generation

Outlines automatically converts model outputs into your desired format.

#### Sync
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

#### Async
```python
import asyncio
from typing import Literal
from pydantic import BaseModel

async def main():
    # Multiple choice (via enum in JSON schema)
    sentiment = await model.generate(
        "The movie was absolutely fantastic!",
        output_type=Literal["positive", "negative", "neutral"]
    )

    # Structured object (Pydantic model)
    class Person(BaseModel):
        name: str
        age: int
        occupation: str

    person = await model.generate(
        "Generate information about a software engineer",
        output_type=Person
    )
```

- Structured outputs may be **unavailable** for `codestral-mamba` models (they lack JSON schema support).
- For best results with complex/nested schemas, use `mistral-large-latest` or higher; simpler models may require retries.
- Check model compatibility:
```python
if model.supports_structured_output():
    # Safe to use Pydantic/JSON schemas
    pass
else:
    print("Use text-only generation")
```

---

### 3.3 Streaming

Streaming methods (`generate_stream` / async `generate_stream`) allow you to iterate over tokens as they are generated.

#### Sync
```python
for chunk in model.generate_stream("Write a short story about AI"):
    print(chunk, end="", flush=True)
```

#### Async
```python
async for chunk in model.generate_stream("Write a short story about AI"):
    print(chunk, end="", flush=True)
```

### 3.4 Structured Output with Streaming

Structured JSON is streamed as text chunks that can be assembled and parsed:

#### Sync Structured Streaming
```python
import json
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

model = from_mistral(mistral_client, "mistral-large-latest")
prompt = """Generate a JSON object representing a person with name and age.
Return only the JSON, no other text."""

chunks = []
for chunk in model.generate_stream(prompt, output_type=Person):
    chunks.append(chunk)

# Assemble and parse the complete JSON
result = "".join(chunks)
parsed = json.loads(result)
print(f"Name: {parsed['name']}, Age: {parsed['age']}")
```

#### Async Structured Streaming
```python
import asyncio
import json
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

async def main():
    model = from_mistral(mistral_client, "mistral-large-latest", async_client=True)
    prompt = """Generate a JSON object representing a person with name and age.
    Return only the JSON, no other text."""

    chunks = []
    async for chunk in model.generate_stream(prompt, output_type=Person):
        chunks.append(chunk)

    # Assemble and parse the complete JSON
    result = "".join(chunks)
    parsed = json.loads(result)
    print(f"Name: {parsed['name']}, Age: {parsed['age']}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Note:** With structured outputs, individual chunks may not be valid JSON until the complete response is assembled.


### 3.5 Batch Processing

Batch inference is currently **not supported** by the `mistralai` library:

```python
# This will raise NotImplementedError
# model.generate_batch(prompts)

# Use sequential processing instead
results = [model.generate(prompt) for prompt in prompts]
```

### 3.6 Error Handling

Usage of this component can raise specific exceptions for common issues:

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

---

## 4. Advanced Usage

### 4.0 Custom Inference Parameters

Pass inference parameters directly to `generate`:

```python
# Sync
result = model.generate(
    "Write a poem",
    temperature=0.9,  # More creative
    max_tokens=200,
    top_p=0.95
)

# Async
result = await model.generate(
    "Write a poem",
    temperature=0.9,
    max_tokens=200,
    top_p=0.95
)
```

### 4.2 Multiple Samples

Generate multiple responses using the `n` parameter:

```python
# Sync
results = model.generate("Respond with one word.", n=2)
assert isinstance(results, list)
assert len(results) == 2

# Async
results = await model.generate("Respond with one word.", n=2)
assert isinstance(results, list)
assert len(results) == 2
```

### 4.3 Multimodal Inputs (Images)

Mistral supports image inputs via `content` lists (e.g., for `pixtral-*` model). Outlines handles this in `Chat` or list inputs:

#### Sync Image Processing
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

#### Async Image Processing
```python
import asyncio
from outlines.inputs import Image
from PIL import Image as PILImage

async def main():
    # Load image
    pil_image = PILImage.open("path/to/image.jpg")
    image = Image(pil_image)

    # Text + image prompt
    result = await model.generate(
        ["Describe this image", image],
        model_name="pixtral-12b-2409"
    )

    # Or via Chat
    chat = Chat([
        {"role": "user", "content": ["What do you see?", image]}
    ])
    result = await model.generate(chat)
```

**Note:** The first item must be a string, followed by optional `Image` objects.
Text-only models will raise errors.

### 4.4 Chat Messages

Use `Chat` objects for multi-turn conversations:

#### Sync Chat
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

#### Async Chat
```python
import asyncio
from outlines.inputs import Chat

async def main():
    chat = Chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."},
        {"role": "user", "content": "Tell me more about its uses."}
    ])

    response = await model.generate(chat)
```

**Supported roles:** `system`, `user`, `assistant`. Other roles raise `ValueError`.

### 4.5 Complex Nested Structures

Based on integration tests, Outlines handles complex nested Pydantic models:

#### Sync Complex Structure
```python
import json
from pydantic import BaseModel
from typing import List
from outlines.inputs import Chat

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Employee(BaseModel):
    name: str
    role: str
    department: str
    address: Address

class Company(BaseModel):
    company_name: str
    employees: List[Employee]

model = from_mistral(mistral_client, "mistral-large-latest")
prompt = Chat([
    {"role": "system", "content": "You are a business data generator. All employees must work in the IT department."},
    {"role": "user", "content": "Generate a company with at least two employees. Return only JSON."}
])

result = model.generate(prompt, output_type=Company)
parsed = json.loads(result)
print(f"Company: {parsed['company_name']}")
for emp in parsed['employees']:
    print(f"Employee: {emp['name']} - {emp['role']}")
```

#### Async Complex Structure
```python
import asyncio
import json

async def main():
    # Same model definitions as above
    model = from_mistral(mistral_client, "mistral-large-latest", async_client=True)
    prompt = Chat([
        {"role": "system", "content": "You are a business data generator. All employees must work in the IT department."},
        {"role": "user", "content": "Generate a company with at least two employees. Return only JSON."}
    ])

    result = await model.generate(prompt, output_type=Company)
    parsed = json.loads(result)
    print(f"Company: {parsed['company_name']}")
    for emp in parsed['employees']:
        print(f"Employee: {emp['name']} - {emp['role']}")
```

---

## 5. Troubleshooting

- **ImportError: No module named 'mistralai'**
  → Run `pip install mistralai`.

- **Authentication Error**
  → Verify `MISTRAL_API_KEY` is set and valid. Test with the [Mistral Playground](https://chat.mistral.ai).

- **Schema Error (e.g., "Mistral does not support your schema")**
  → Avoid regex/CFG; ensure no `pattern` fields in Pydantic (Outlines sets `additionalProperties: false`); try a simpler schema or different model.

- **Model Not Found Error**
  → Confirm the model name (e.g., `"mistral-small-latest"`) and your subscription tier. Check [docs](https://docs.mistral.ai/getting-started/models/).

- **Rate Limits or Quotas**
  → Monitor usage in the Mistral console; upgrade your plan for higher limits.

- **Input Validation Errors**
  → Ensure Chat messages use valid roles (`system`, `user`, `assistant`); list inputs start with strings.

---

## 6. Notes

- Outlines automatically sets `additionalProperties: false` in JSON schemas for stricter parsing.
- For multimodal inputs, use compatible models
- `generate_batch` raises `NotImplementedError` due to API limitations.
- For more examples (e.g., vision classification), see the integration tests at tests/models/test_mistral.py
- **Author:** Steven E. Elliott aka see (seeyallc6c@gmail.com)

---

*Last updated: September 19, 2025*
