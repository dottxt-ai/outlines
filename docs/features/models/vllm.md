---
title: vLLM
---

# vLLM

## Prerequisites

The Outlines `VLLM` model is intended to be used along with a vLLM instance running on a separate server (can be local or remote). Make sure you have a vLLM server running and accessible before using the `VLLM` model. For instance by running:

```shell
pip install vllm

vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

Follow the [Installation instructions](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) for more information on how to set up a vLLM server for your particular setup.

As the vLLM client relies on the `openai` python sdk, you need to have the `openai` package installed. Install all optional dependencies for the `VLLM` model with: `pip install openai`.

If you want to use the vllm offline inference mode instead of the server mode, please refer to the [VLLMOffline](./vllm_offline.md) model documentation.

## Model Initialization

To load the model, you can use the `from_vllm` function. The argument of the function is either an `OpenAI` or `AsyncOpenAI` instance from the `openai` library. Make sure the value of the `base_url` argument of the `OpenAI` client points to your running vLLM server. Consult the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) on using an OpenAI client with a vLLM server for more information.

Based on whether the `openai` client instance is synchronous or asynchronous, you will receive a `VLLM` or `AsyncVLLM` model instance.

For instance:

```python
import openai
import outlines

# Create the OpenAI client
sync_openai_client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
async_openai_client = openai.AsyncOpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")

# Create a sync model
sync_model = outlines.from_vllm(sync_openai_client)
print(type(sync_model)) # <class 'outlines.models.vllm.VLLM'>

# Create an async model
async_model = outlines.from_vllm(async_openai_client)
print(type(async_model)) # <class 'outlines.models.vllm.AsyncVLLM'>
```

## Text Generation

To generate text, you can simply call the model with a prompt.

For instance:

```python
import openai
import outlines

# Create the model
model = outlines.from_openai(openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct"))

# Call it to generate text
response = model("What's the capital of Latvia?", max_tokens=20)
print(response) # 'Riga'
```

#### Vision

Some models you can run with VLLM support vision input. To use this feature, provide a list containing a text prompt and `Image` instances.

For instance:

```python
import io
import requests
import PIL
import outlines
import openai
from outlines.inputs import Image

# Create the model
model = outlines.from_openai(openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct"))

# Function to get an image
def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# Create the prompt containing the text and the image
prompt = [
    "Describe the image",
    Image(get_image("https://picsum.photos/id/237/400/300"))
]

# Call the model to generate a response
response = model(prompt, max_tokens=50)
print(response) # 'This is a picture of a black dog.'
```

#### Chat

You can also use chat inputs with the `VLLM` model. To do so, call the model with a `Chat` instance. The content of messsage within the chat can be vision inputs as described above.

For instance:

```python
import io
import requests
import PIL
import openai
import outlines
from outlines.inputs import Chat, Image

# Create the model
model = outlines.from_openai(openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct"))

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
response = model(prompt, max_tokens=50)
print(response) # 'This is a picture of a black dog.'
```

#### Streaming

Finally, the `VLLM` model supports streaming through the `stream` method.

For instance:

```python
import openai
import outlines

# Create the model
model = outlines.from_openai(openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct"))

# Stream the response
for chunk in model.stream("Tell me a short story about a cat.", max_tokens=50):
    print(chunk) # 'Once...'
```

## Asynchronous Calls

vLLM supports asynchronous operations by passing an `AsyncOpenAI` client instead of a regular `OpenAI` client. This returns an `AsyncVLLM` model instance that supports async/await patterns.

### Basic Async Generation

```python
import asyncio
import openai
import outlines

async def generate_text():
    async_client = openai.AsyncOpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
    async_model = outlines.from_vllm(async_client)

    result = await async_model("Write a haiku about Python.", max_tokens=50)
    print(result)

asyncio.run(generate_text())
```

### Async Streaming

The async model also supports streaming with async iteration:

```python
import asyncio
import openai
import outlines

async def stream_text():
    async_client = openai.AsyncOpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
    async_model = outlines.from_vllm(async_client)

    async for chunk in async_model.stream("Tell me a story about a robot.", max_tokens=100):
        print(chunk, end="")

asyncio.run(stream_text())
```

### Concurrent Async Requests

One of the main benefits of async calls is the ability to make multiple concurrent requests:

```python
import asyncio
import openai
import outlines

async def generate_multiple():
    async_client = openai.AsyncOpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
    async_model = outlines.from_vllm(async_client)

    prompts = [
        "Write a tagline for a coffee shop.",
        "Write a tagline for a bookstore.",
        "Write a tagline for a gym."
    ]

    tasks = [async_model(prompt, max_tokens=30) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, result in zip(prompts, results):
        print(f"{prompt}\n{result}\n")

asyncio.run(generate_multiple())
```

## Structured Generation

vLLM supports all output types available in Outlines. Simply provide an `output_type` after the prompt when calling the model. All structured generation features work with both synchronous and asynchronous models.

### Simple Type

```python
import openai
import outlines

output_type = int

openai_client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
model = outlines.from_vllm(openai_client)

result = model("How many countries are there in the world?", output_type)
print(result) # '200'
```

### JSON Schema

```python
import openai
import outlines
from typing import List
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

openai_client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
model = outlines.from_vllm(openai_client)

result = model("Create a character.", output_type=Character, frequency_penalty=1.5)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

### Multiple Choice

```python
from typing import Literal
import openai
import outlines

output_type = Literal["Paris", "London", "Rome", "Berlin"]

openai_client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
model = outlines.from_vllm(openai_client)

result = model("What is the capital of France?", output_type, temperature=0)
print(result) # 'Paris'
```

### Regex

```python
import openai
import outlines
from outlines.types import Regex

output_type = Regex(r"\d{3}-\d{2}-\d{4}")

openai_client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
model = outlines.from_vllm(openai_client)

result = model("Generate a fake social security number.", output_type, top_p=0.1)
print(result) # '782-32-3789'
```

### Context-Free Grammar

```python
import openai
import outlines
from outlines.types import CFG

arithmetic_grammar = """
?start: sum

?sum: product
| sum "+" product   -> add
| sum "-" product   -> sub

?product: atom
| product "*" atom  -> mul
| product "/" atom  -> div

?atom: NUMBER           -> number
| "-" atom         -> neg
| "(" sum ")"

%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""
output_type = CFG(arithmetic_grammar)

openai_client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
model = outlines.from_vllm(openai_client)

result = model("Write an addition.", output_type, extra_body={"guided_decoding_backend": "outlines"})
print(result) # '23 + 48'
```

### Async Structured Generation

All structured generation features work seamlessly with async models:

```python
import asyncio
import openai
import outlines
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int

async def generate_user():
    async_client = openai.AsyncOpenAI(base_url="http://0.0.0.0:8000/v1", "microsoft/Phi-3-mini-4k-instruct")
    async_model = outlines.from_vllm(async_client)

    result = await async_model("Generate a random user profile.", output_type=User)
    user = User.model_validate_json(result)
    print(f"Name: {user.name}, Email: {user.email}, Age: {user.age}")

asyncio.run(generate_user())
```

## Inference Arguments

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `chat.completions.create` method of the OpenAI client.

An optional parameter of particular interest is `extra_body`, which is a dictionary containing arguments that are specific to vLLM and are not part of the standard `openai` interface. Among those, `guided_decoding_backend` allows you to select the library used by the vLLM server to control structured generation. You can use the value `outlines` to generated structured text with Outlines.

See the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters) on extra parameters for the OpenAI-compatible server for more information on inference parameters.
