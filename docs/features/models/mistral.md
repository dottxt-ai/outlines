# Mistral AI

Outlines supports models from Mistral AI's API. You'll need to install the `mistralai` package and set up your API key.

## Installation

```bash
pip install mistralai
```

## Basic Usage

```python
import os
import outlines

# Set your API key
os.environ["MISTRAL_API_KEY"] = "your-api-key-here"

# Initialize the model
model = outlines.models.mistral("mistral-large-latest")

# Generate text
result = model("What is the capital of France?")
print(result)
```

## Available Models

Mistral AI offers several models through their API:

- `mistral-tiny` - Fastest model for simple tasks
- `mistral-small` - Balanced speed and capability
- `mistral-medium` - More capable model
- `mistral-large` - Most capable model
- `mistral-large-latest` - Latest version of the large model
- `open-mistral-7b` - Open source 7B model
- `open-mixtral-8x7b` - Open source 8x7B mixture of experts
- `open-mixtral-8x22b` - Open source 8x22B mixture of experts

## Configuration

You can configure various parameters when creating the model:

```python
model = outlines.models.mistral(
    "mistral-large-latest",
    api_key="your-api-key",
    system_prompt="You are a helpful assistant specialized in data analysis.",
    config={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1000
    }
)
```

## Structured Generation

Use Mistral models with Outlines' structured generation features:

```python
from typing import Literal
from pydantic import BaseModel

model = outlines.models.mistral("mistral-large-latest")

# Multiple choice
sentiment = model(
    "The movie was absolutely fantastic!",
    Literal["positive", "negative", "neutral"]
)

# Structured objects
class Person(BaseModel):
    name: str
    age: int
    occupation: str

person = model(
    "Generate information about a software engineer",
    Person
)
```

## Streaming

Mistral models support streaming generation:

```python
model = outlines.models.mistral("mistral-large-latest")

for chunk in model.stream("Write a short story about AI"):
    print(chunk, end="", flush=True)
```

## Batch Processing

You can process multiple prompts at once:

```python
model = outlines.models.mistral("mistral-large-latest")

prompts = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]

results = model(prompts)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

## Error Handling

The Mistral integration includes proper error handling for API issues:

```python
try:
    model = outlines.models.mistral("mistral-large-latest", api_key="invalid-key")
    result = model("Hello")
except RuntimeError as e:
    print(f"API Error: {e}")
```

## Environment Variables

You can set your Mistral API key using environment variables:

```bash
export MISTRAL_API_KEY="your-api-key-here"
```

Then use the model without explicitly passing the API key:

```python
model = outlines.models.mistral("mistral-large-latest")
```

## Advanced Usage

### Custom System Prompts

```python
model = outlines.models.mistral(
    "mistral-large-latest",
    system_prompt="You are an expert Python programmer. Always provide clean, efficient code with explanations."
)

code = model("Write a function to calculate fibonacci numbers")
```

### Temperature and Randomness Control

```python
# More deterministic responses
model_deterministic = outlines.models.mistral(
    "mistral-large-latest",
    config={"temperature": 0.1, "top_p": 0.1}
)

# More creative responses  
model_creative = outlines.models.mistral(
    "mistral-large-latest",
    config={"temperature": 0.9, "top_p": 0.95}
)
```

## Rate Limits and Best Practices

- Mistral AI has rate limits on their API. Handle rate limit errors appropriately.
- Use batch processing when possible to reduce the number of API calls.
- Consider using smaller models (mistral-small, mistral-tiny) for simpler tasks to reduce costs.
- Cache results when appropriate to avoid redundant API calls.

## Troubleshooting

**ImportError: No module named 'mistralai'**
Install the Mistral AI package: `pip install mistralai`

**Authentication Error**
Make sure your API key is set correctly in the environment or passed to the model.

**Rate Limit Error**
Implement exponential backoff or reduce the frequency of your requests.

**Model Not Found Error**  
Verify that the model name is correct and available in your Mistral AI plan.