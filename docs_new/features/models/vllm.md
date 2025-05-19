---
title: vLLM
---

# vLLM

## Prerequisites

The Outlines `VLLM` model is intended to be used along with a vLLM instance running on a separate server (can be local or remote). Make sure you have a vLLM server running and accessible before using the `VLLM` model. As the vLLM client relies on the `openai` python sdk, you need to have an `openai` package installed: `pip install openai`.

If you want to use the vllm offline inference mode instead of the server mode, please refer to the [VLLMOffline](./vllm_offline.md) model documentation.

## Model Initialization

To load the model, you can use the `from_vllm` function. The argument of the function is either an `OpenAI` or `AsyncOpenAI` instance from the `openai` library. Make sure the value of the `base_url` argument of the `OpenAI` client points to your running vLLM server. Consult the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) on using an OpenAI client with a vLLM server for more information.

Based on whether the `openai` client instance is synchronous or asynchronous, you will receive a `VLLM` or `AsyncVLLM` model instance.

For instance:

```python
import openai
import outlines

# Create the OpenAI client
sync_openai_client = openai.OpenAI(base_url="http://localhost:8080")
async_openai_client = openai.AsyncOpenAI(base_url="http://localhost:8080")

# Create a sync model
sync_model = outlines.from_vllm(sync_openai_client)
print(type(sync_model)) # <class 'outlines.models.vllm.VLLM'>

# Create an async model
async_model = outlines.from_vllm(async_openai_client)
print(type(async_model)) # <class 'outlines.models.vllm.AsyncVLLM'>
```

## Text Generation

To generate text, you can simply call the model with a prompt. The `VLLM` model supports streaming.

For instance:

```python
import openai
import outlines

openai_client = openai.OpenAI(base_url="http://localhost:8080")

model = outlines.from_vllm(openai_client)

for chunk in sync_model.stream("Write a short story about a cat.", max_tokens=100):
    print(chunk)
```

## Structured Generation

vLLM supports all output types available in Outlines. Simply provide an `output_type` after the prompt when calling the model.

### JSON Schema

```python
import openai
import outlines
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

openai_client = openai.OpenAI(base_url="http://localhost:8080")
model = outlines.from_vllm(openai_client)

result = model("Create a character.", output_type=Character, frequency_penalty=1.5)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
```

### Multiple Choice

```python
from typing import Literal
import openai
import outlines

output_type = Literal["Paris", "London", "Rome", "Berlin"]

openai_client = openai.OpenAI(base_url="http://localhost:8080")
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

openai_client = openai.OpenAI(base_url="http://localhost:8080")
model = outlines.from_vllm(openai_client)

result = model("Generate a fake social security number.", output_type, top_p=0.1)
print(result) # 782-32-3789
```

### Context-Free Grammar

```python
import openai
import outlines
from outlines.text import CFG

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

openai_client = openai.OpenAI(base_url="http://localhost:8080")
model = outlines.from_vllm(openai_client)

result = sync_model("Write an addition.", output_type, extra_body={"guided_decoding_backend": "outlines"})
print(result) # '23 + 48'
```

## Inference Parameters

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `openai` client.

An optional parameter of particular interest is `extra_body`, which is a dictionnary containing arguments that are specific to vLLM and are not part of the standard `openai` interface. Among those, `guided_decoding_backend` allows you to select the library used by the vLLM server to control structured generation. You can use the value `outlines` to generated structured text with Outlines.

See the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters) on extra parameters for the OpenAI-compatible server for more information on inference parameters.
