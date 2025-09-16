
# Mistral AI Integration
Outlines now supports models from [Mistral AI](https://mistral.ai) via the official `mistralai` package.
This integration provides a thin wrapper around the `mistralai.Mistral` client, handling prompt formatting, structured outputs, and streaming.

## Installation

```bash
pip install mistralai
```
Basic Usage

```python
import os
import outlines
from mistralai import Mistral as MistralClient

# ...

# Set your API key
os.environ["MISTRAL_API_KEY"] = "your-api-key-here"
# or in terminal: export MISTRAL_API_KEY=YourRawKeyHere

# Initialize the client and model
client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
model = outlines.models.mistral.from_mistral(client, "mistral-large-latest")

# Generate text
result = model.generate("What is the capital of France?")
print(result)
```
## Available Models
Mistral AI offers several models:
 - mistral-tiny
 - mistral-small
 - mistral-medium
 - mistral-large
 - mistral-large-latest
 - open-mistral-7b
 - open-mixtral-8x7b
 - open-mixtral-8x22b
 - codestral-mamba (⚠ does not support structured outputs, AVOID)

## Configuration
You can set default inference parameters when creating the model:
```python
model = outlines.models.mistral.from_mistral(
    client,
    "mistral-large-latest",
    system_prompt="You are a helpful assistant specialized in data analysis.",
    config={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1000
    }
)
```
## Structured Generation
Mistral supports structured JSON outputs. Outlines automatically converts Pydantic models, dataclasses, and TypedDicts into JSON schemas.

```python
from typing import Literal
from pydantic import BaseModel

# Multiple choice
sentiment = model.generate(
    "The movie was absolutely fantastic!",
    Literal["positive", "negative", "neutral"]
)

# Structured object
class Person(BaseModel):
    name: str
    age: int
    occupation: str

person = model.generate(
    "Generate information about a software engineer",
    Person
)

```
 ⚠️ Note: Regex and CFG-based structured outputs are not currently supported by the mistralai API.

## Streaming
Stream responses chunk by chunk:

```python
for chunk in model.generate_stream("Write a short story about AI"):
    print(chunk, end="", flush=True)
```
## Batch Processing
Batch inference is currently not supported by the mistralai API.
Please use multiple calls to generate if you need to process a list of prompts.

## Error Handling
The Mistral integration raises helpful exceptions for schema issues or API errors:
```python
try:
    result = model.generate("Hello")
except RuntimeError as e:
    print(f"API Error: {e}")
except TypeError as e:
    print(f"Schema Error: {e}")
```from_mistral(client, "mistral-large-latest")
```

## Advanced Usage
### Custom System Prompts
``` python
model = from_mistral(
    client,
    "mistral-large-latest",
    system_prompt="You are an expert Python programmer."
)

code = model.generate("Write a function to calculate fibonacci numbers")
```
### Temperature and Randomness Control
``` python
# Deterministic
det_model = from_mistral(
    client,
    "mistral-large-latest",
    config={"temperature": 0.1, "top_p": 0.1}
)

# Creative
creative_model = from_mistral(
    client,
    "mistral-large-latest",
    config={"temperature": 0.9, "top_p": 0.95}
)
```

## Troubleshooting
***ImportError: No module named 'mistralai'***
→ Install with pip install mistralai

***Authentication Error***
→ Ensure your API key is set correctly

***Schema Error***
→ Regex and CFG-based schemas are not supported by Mistral

***Model Not Found Error***
→ Check that the model name is valid and available in your plan

## Notes
Structured outputs are supported for all models except codestral-mamba.

generate_batch is not available due to API limitations.

Outlines automatically disables additionalProperties in JSON schemas for stricter outputs.

Author: Steven E. Elliott aka see (seeyallc6c@gmail.com)

---
