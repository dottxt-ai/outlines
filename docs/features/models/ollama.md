---
title: Ollama
---

# Ollama

!!! Installation

    To be able to use Ollama in Outlines, you must install both Ollama and the optional dependency libraries of the model.

    - To download Ollama: https://ollama.com/download
    - To install the ollama python sdk: `pip install outlines[ollama]`

    Consult the [`ollama` documentation](https://github.com/ollama/ollama-python) for detailed information on installation and client initialization.

## Model Initialization

To create an Ollama model instance, you can use the `from_ollama` function. It takes 2 arguments:

- `client`: an `ollama.Client` or `ollama.AsyncClient` instance
- `model_name`: the name of the model you want to use

Based on whether the inference client instance is synchronous or asynchronous, you will receive an `Ollama` or an `AsyncOllama` model instance.

For instance:

```python
import ollama
import outlines

# Create the client or async client
client = ollama.Client()
async_client = ollama.AsyncClient()

# Create a sync model
model = outlines.from_ollama(
    client,
    "qwen2.5vl:3b",
)

# Create an async model
model = outlines.from_ollama(
    async_client,
    "qwen2.5vl:3b",
)
```

You can find the list of available models on the [Ollama library](https://ollama.com/library).

## Text Generation

Once you've created your Outlines `Ollama` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt.

For instance:

```python
import ollama
import outlines

# Create the model
model = outlines.from_ollama(ollama.Client(), "qwen2.5vl:3b")

# Call it to generate text
response = model("What's the capital of Latvia?")
print(response) # 'Riga'
```

Ollama also supports streaming. For instance:

```python
import ollama
import outlines

# Create the model
model = outlines.from_ollama(ollama.Client(), "qwen2.5vl:3b")

# Stream text
for chunk in model.stream("Write a short story about a cat"):
    print(chunk) # 'In...'
```

Additionally, you can use `Ollama` with the `Vision` input if you're running a vision model such as qwen2.5vl. For instance:

```python
import io
import requests
import PIL
import ollama
import outlines
from outlines.templates import Vision

# Create the model
model = outlines.from_ollama(
    ollama.Client(),
    "qwen2.5vl:3b"
)

# Function to get an image
def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# Create the prompt
prompt = Vision("Describe the image", get_image("https://picsum.photos/id/237/400/300"))

# Generate text
response = model(prompt)
print(response) # The image shows a black puppy with a curious and attentive expression.
```

## Asynchronous Calls

Ollama supports asynchronous operations by passing an `AsyncClient` instead of a regular `Client`. This returns an `AsyncOllama` model instance that supports async/await patterns.

### Basic Async Generation

```python
import asyncio
import outlines
import ollama

async def generate_text():
    # Create an async model
    async_client = ollama.AsyncClient()
    async_model = outlines.from_ollama(async_client, "qwen2.5vl:3b")

    result = await async_model("Write a haiku about Python.")
    print(result)

asyncio.run(generate_text())
```

### Async Streaming

The async model also supports streaming with async iteration:

```python
import asyncio
import outlines
import ollama

async def stream_text():
    async_client = ollama.AsyncClient()
    async_model = outlines.from_ollama(async_client, "qwen2.5vl:3b")

    async for chunk in async_model.stream("Tell me a story about a robot."):
        print(chunk, end="")

asyncio.run(stream_text())
```

### Concurrent Async Requests

One of the main benefits of async calls is the ability to make multiple concurrent requests:

```python
import asyncio
import outlines
import ollama

async def generate_multiple():
    async_client = ollama.AsyncClient()
    async_model = outlines.from_ollama(async_client, "qwen2.5vl:3b")

    # Define multiple prompts
    prompts = [
        "Write a tagline for a coffee shop.",
        "Write a tagline for a bookstore.",
        "Write a tagline for a gym."
    ]

    tasks = [async_model(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, result in zip(prompts, results):
        print(f"{prompt}\n{result}\n")

asyncio.run(generate_multiple())
```

## Structured Generation

Ollama only provides support for structured generation based on a JSON schema. To use it, call the model with a JSON schema object as an `output_type` on top of your prompt.

For instance:

```python
from typing import List
from pydantic import BaseModel
import ollama
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_ollama(ollama.Client(), "tinyllama")

# Call it with the output type to generate structured text
result = model("Create a character", Character)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

## Inference arguments

When calling the model, you can provide keyword arguments that will be passed down to the `generate` method of the Ollama client.

Consult the [Ollama REST API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) for the full list of inference parameters.
