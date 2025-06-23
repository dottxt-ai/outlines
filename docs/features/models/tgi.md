---
title: TGI
---

# TGI

## Prerequisites

The Outlines `TGI` model is intended to be used along with a HuggingFace `Text Generation Inference` server (running locally or remotely). Make sure you have a TGI server running before using the `TGI` model. For instance running:

```shell
docker run \
  --gpus all \
  --shm-size 1g \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:3.3.4 \
  --model-id NousResearch/Meta-Llama-3-8B-Instruct
```

Please consult the [installation guide](https://huggingface.co/docs/text-generation-inference/en/quicktour) for more information about how to run TGI with your particular setup.
As the TGI client relies on the `huggingface_hub` python package, you need to have it installed. Install all optional dependencoes of the `TGI` model with: `pip install outlines[tgi]`

## Model Initialization

To load the model, you can use the `from_tgi` function. The argument of the function is either an `InferenceClient` or `AsyncInferenceClient` instance from the `huggingface_hub` library. Consult the [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client) for more information on their inference client.

Based on whether the inference client instance is synchronous or asynchronous, you will receive a `TGI` or an `AsyncTGI` model instance.

For instance:

```python
import outlines
import huggingface_hub

# Create the inference client
client = huggingface_hub.InferenceClient("http://localhost:11434")
async_client = huggingface_hub.AsyncInferenceClient("http://localhost:11434")

# Create a sync model
sync_model = outlines.from_tgi(client)
print(type(sync_model))  # <class 'outlines.models.tgi.TGI'>

# Create an async model
async_model = outlines.from_tgi(async_client)
print(type(async_model))  # <class 'outlines.models.tgi.AsyncTGI'>
```

## Text Generation

To generate text, you can simply call the model with a prompt.

For instance:

```python
import outlines
import huggingface_hub

# Create the model
client = huggingface_hub.InferenceClient("http://localhost:11434")
model = outlines.from_tgi(client)

# Call it to generate text
result = model("Write a short story about a cat.", stop_sequences=["."])
print(result) # 'In a quiet village where the cobblestones hummed softly beneath the morning mist...'
```

The `TGI` model supports streaming. For instance:

```python
import outlines
import huggingface_hub

# Create the model
client = huggingface_hub.InferenceClient("http://localhost:11434")
model = outlines.from_tgi(client)

# Stream text
for chunk in model.stream("Write a short story about a cat.", stop_sequences=["."]):
    print(chunk) # 'In ...'
```

## Asynchronous Calls

TGI supports asynchronous operations by passing an `AsyncInferenceClient` instead of a regular `InferenceClient`. This returns an `AsyncTGI` model instance that supports async/await patterns.

### Basic Async Generation

```python
import asyncio
import outlines
import huggingface_hub

async def generate_text():
    # Create an async model
    async_client = huggingface_hub.AsyncInferenceClient("http://localhost:11434")
    async_model = outlines.from_tgi(async_client)

    result = await async_model("Write a haiku about Python.", max_new_tokens=50)
    print(result)

asyncio.run(generate_text())
```

### Async Streaming

The async model also supports streaming with async iteration:

```python
import asyncio
import outlines
import huggingface_hub

async def stream_text():
    async_client = huggingface_hub.AsyncInferenceClient("http://localhost:11434")
    async_model = outlines.from_tgi(async_client)

    async for chunk in async_model.stream("Tell me a story about a robot.", max_new_tokens=100):
        print(chunk, end="")

asyncio.run(stream_text())
```

### Concurrent Async Requests

One of the main benefits of async calls is the ability to make multiple concurrent requests:

```python
import asyncio
import outlines
import huggingface_hub

async def generate_multiple():
    async_client = huggingface_hub.AsyncInferenceClient("http://localhost:11434")
    async_model = outlines.from_tgi(async_client)

    # Define multiple prompts
    prompts = [
        "Write a tagline for a coffee shop.",
        "Write a tagline for a bookstore.",
        "Write a tagline for a gym."
    ]

    tasks = [async_model(prompt, max_new_tokens=30) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, result in zip(prompts, results):
        print(f"{prompt}\n{result}\n")

asyncio.run(generate_multiple())
```

## Structured Generation

TGI supports all output types available in Outlines except for context-free grammars. Simply provide an `output_type` after the prompt when calling the model. All structured generation features work with both synchronous and asynchronous models.

### Simple Type

```python
import outlines
import huggingface_hub

output_type = int

tgi_client = huggingface_hub.InferenceClient("http://localhost:8080")
model = outlines.from_tgi(tgi_client)

result = model("How many countries are there in the world?", output_type)
print(result) # '200'
```### JSON Schema

```python
import outlines
import huggingface_hub
from typing import List
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

tgi_client = huggingface_hub.InferenceClient("http://localhost:8080")
model = outlines.from_tgi(tgi_client)

result = model("Create a character.", output_type=Character, frequency_penalty=1.5)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```### Multiple Choice

```python
import outlines
import huggingface_hub
from typing import Literal

output_type = Literal["Paris", "London", "Rome", "Berlin"]

tgi_client = huggingface_hub.InferenceClient("http://localhost:8080")
model = outlines.from_tgi(tgi_client)

result = model("What is the capital of France?", output_type, temperature=0)
print(result) # 'Paris'
```### Regex

```python
import outlines
import huggingface_hub
from outlines.types import Regex

output_type = Regex(r"\d{3}-\d{2}-\d{4}")

tgi_client = huggingface_hub.InferenceClient("http://localhost:8080")
model = outlines.from_tgi(tgi_client)

result = model("Generate a fake social security number.", output_type, top_p=0.1)
print(result) # '782-32-3789'
```

### Async Structured Generation

All structured generation features work seamlessly with async models:

```python
import asyncio
import outlines
import huggingface_hub
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int

async def generate_user():
    async_client = huggingface_hub.AsyncInferenceClient("http://localhost:11434")
    async_model = outlines.from_tgi(async_client)

    result = await async_model("Generate a random user profile.", output_type=User)
    user = User.model_validate_json(result)
    print(f"Name: {user.name}, Email: {user.email}, Age: {user.age}")

asyncio.run(generate_user())
```

## Inference parameters

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `text_generation` method of the TGI client.

Common parameters include `max_new_tokens`, `stop_sequences`, `temperature`, `top_k`, `top_p`, and others as specified in the [TGI inference client documentation](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient).
