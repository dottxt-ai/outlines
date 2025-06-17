---
title: TGI
---

# TGI

## Prerequisites

The Outlines `TGI` model is intended to be used along with a HuggingFace `Text Generation Inference` server (running locally or remotely). Make sure you have a TGI server running before using the `TGI` model. As the TGI client relies on the `huggingface_hub` python package, you need to have it installed. Install all optional dependencoes of the `TGI` model with: `pip install outlines[tgi]`

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
```The `TGI` model supports streaming. For instance:

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

## Structured Generation

TGI supports all output types available in Outlines except for context-free grammars. Simply provide an `output_type` after the prompt when calling the model.

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
## Inference parameters

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `text_generation` method of the TGI client.

Common parameters include `max_new_tokens`, `stop_sequences`, `temperature`, `top_k`, `top_p`, and others as specified in the [TGI inference client documentation](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient).
