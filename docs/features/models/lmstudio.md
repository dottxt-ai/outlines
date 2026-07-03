---
title: LM Studio
---

# LM Studio

!!! Installation

    To be able to use LM Studio in Outlines, you must install both LM Studio and the optional dependency libraries of the model.

    - To download LM Studio: https://lmstudio.ai/download
    - To install the lmstudio python sdk: `pip install "outlines[lmstudio]"`

    Consult the [`lmstudio-python` documentation](https://lmstudio.ai/docs/python) for detailed information on installation and client initialization.

## Model Initialization

To create an LM Studio model instance, you can use the `from_lmstudio` function. It takes 2 arguments:

- `client`: an `lmstudio.Client` or `lmstudio.AsyncClient` instance
- `model_name`: the name of the model you want to use (optional; if not provided, the default model loaded in LM Studio is used)

Based on whether the inference client instance is synchronous or asynchronous, you will receive an `LMStudio` or an `AsyncLMStudio` model instance.

For instance:

```python
import lmstudio
import outlines

# Create the client or async client
client = lmstudio.Client()
async_client = lmstudio.AsyncClient()

# Create a sync model
model = outlines.from_lmstudio(
    client,
    "openai/gpt-oss-20b",
)

# Create an async model
model = outlines.from_lmstudio(
    async_client,
    "openai/gpt-oss-20b",
)
```

You can find the list of available models in the [LM Studio model catalog](https://lmstudio.ai/models). The model must be downloaded in LM Studio (through the app or with `lms get <model_name>`) before you can use it.

## Text Generation

Once you've created your Outlines `LMStudio` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt.

For instance:

```python
import lmstudio
import outlines

# Create the model
model = outlines.from_lmstudio(lmstudio.Client(), "openai/gpt-oss-20b")

# Call it to generate text
response = model("What's the capital of Latvia?")
print(response) # 'Riga'
```

#### Vision

Some LM Studio models support vision input. To use this feature, provide a list containing a text prompt and `Image` instances.

```python
import io
import requests
import PIL
import lmstudio
import outlines
from outlines.inputs import Image

# Create the model
model = outlines.from_lmstudio(
    lmstudio.Client(),
    "qwen/qwen2.5-vl-7b"
)

# Function to get an image
def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# Create the prompt
prompt = [
    "Describe the image",
    Image(get_image("https://picsum.photos/id/237/400/300"))
]

# Generate text
response = model(prompt)
print(response) # The image shows a black puppy with a curious and attentive expression.
```

#### Chat

You can also use chat inputs with the `LMStudio` model. To do so, call the model with a `Chat` instance. The content of messages within the chat can be vision inputs as described above.

For instance:

```python
import io
import requests
import PIL
import lmstudio
import outlines
from outlines.inputs import Chat, Image

# Create the model
model = outlines.from_lmstudio(
    lmstudio.Client(),
    "qwen/qwen2.5-vl-7b"
)

# Function to get an image
def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# Create the chat input
prompt = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": ["Describe the image", Image(get_image("https://picsum.photos/id/237/400/300"))]
    },
])

# Call the model to generate a response
response = model(prompt)
print(response) # 'This is a picture of a black dog.'
```

#### Streaming

Finally, the `LMStudio` model supports streaming through the `stream` method.

```python
import lmstudio
import outlines

# Create the model
model = outlines.from_lmstudio(lmstudio.Client(), "openai/gpt-oss-20b")

# Stream text
for chunk in model.stream("Write a short story about a cat"):
    print(chunk) # 'In...'
```

## Asynchronous Calls

LM Studio supports asynchronous operations by passing an `AsyncClient` instead of a regular `Client`. This returns an `AsyncLMStudio` model instance that supports async/await patterns.

### Basic Async Generation

```python
import asyncio
import outlines
import lmstudio

async def generate_text():
    # Create an async model
    async_client = lmstudio.AsyncClient()
    async_model = outlines.from_lmstudio(async_client, "openai/gpt-oss-20b")

    result = await async_model("Write a haiku about Python.")
    print(result)

    # Release the client's resources once you're done
    await async_model.close()

asyncio.run(generate_text())
```

The `AsyncLMStudio` model manages the lifecycle of its underlying async client: the client session is entered on the first call to the model. Once you're done using the model, call its `close` method to release the resources.

### Async Streaming

The async model also supports streaming with async iteration:

```python
import asyncio
import outlines
import lmstudio

async def stream_text():
    async_client = lmstudio.AsyncClient()
    async_model = outlines.from_lmstudio(async_client, "openai/gpt-oss-20b")

    async for chunk in async_model.stream("Tell me a story about a robot."):
        print(chunk, end="")

    await async_model.close()

asyncio.run(stream_text())
```

### Concurrent Async Requests

One of the main benefits of async calls is the ability to make multiple concurrent requests:

```python
import asyncio
import outlines
import lmstudio

async def generate_multiple():
    async_client = lmstudio.AsyncClient()
    async_model = outlines.from_lmstudio(async_client, "openai/gpt-oss-20b")

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

    await async_model.close()

asyncio.run(generate_multiple())
```

## Structured Generation

LM Studio only provides support for structured generation based on a JSON schema. To use it, call the model with a JSON schema object as an `output_type` on top of your prompt.

For instance:

```python
from typing import List
from pydantic import BaseModel
import lmstudio
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_lmstudio(lmstudio.Client(), "openai/gpt-oss-20b")

# Call it with the output type to generate structured text
result = model("Create a character", Character)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

## Inference arguments

When calling the model, you can provide keyword arguments that will be passed down to the `respond` method of the LM Studio model handle. For instance, you can control sampling with the `config` argument:

```python
import lmstudio
import outlines

model = outlines.from_lmstudio(lmstudio.Client(), "openai/gpt-oss-20b")

response = model(
    "What's the capital of Latvia?",
    config={"temperature": 0.5, "maxTokens": 100},
)
```

Consult the [LM Studio prediction parameters documentation](https://lmstudio.ai/docs/python/llm-prediction/parameters) for the full list of inference parameters.
