---
title: SGLang
---

# SGLang

## Prerequisites

The Outlines `SgLang` model is intended to be used along with an SGLang instance running on a separate server (can be local or remote). Make sure you have a SGLang server running and accessible before using the `SgLang` model. As the SGLang client relies on the `openai` python sdk, you need to have an `openai` package installed: `pip install openai`.

When launching your SGLang server, you can specify the backend engine to use for structured generation through the `grammar-backend`. Add `--grammar-backend outlines` to your command to use Outlines instead of the default engine.

## Model Initialization

To load the model, you can use the `from_sglang` function. The argument of the function is either an `OpenAI` or `AsyncOpenAI` instance from the `openai` library. Make sure the value of the `base_url` argument of the `OpenAI` client points to your running SGLang server. Consult the [SGLang documentation](https://docs.sglang.ai/backend/send_request.html) on using an OpenAI client with an SGLang server for more information.

Based on whether the `openai` client instance is synchronous or asynchronous, you will receive a `SgLang` or `AsyncSgLang` model instance.

For instance:

```python
import openai
import outlines

# Create the OpenAI client
sync_openai_client = openai.OpenAI(base_url="http://localhost:11434")
async_openai_client = openai.AsyncOpenAI(base_url="http://localhost:11434")

# Create a sync model
sync_model = outlines.from_sglang(sync_openai_client)
print(type(sync_model)) # <class 'outlines.models.vllm.VLLM'>

# Create an async model
async_model = outlines.from_sglang(async_openai_client)
print(type(async_model)) # <class 'outlines.models.vllm.AsyncVLLM'>
```

## Text Generation

To generate text, you can simply call the model with a prompt. The `SgLang` model supports streaming.

For instance:

```python
import openai
import outlines

openai_client = openai.OpenAI(base_url="http://localhost:11434")
model = outlines.from_vllm(openai_client)

for chunk in sync_model.stream("Write a short story about a cat.", max_tokens=100):
    print(chunk)
```

## Structured Generation

SGLang supports all output types available in Outlines (context-free grammars with caveats though, see the subsection below for more details). Simply provide an `output_type` after the prompt when calling the model.

### JSON Schema

```python
import openai
import outlines
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

openai_client = openai.OpenAI(base_url="http://localhost:11434")
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

openai_client = openai.OpenAI(base_url="http://localhost:11434")
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

openai_client = openai.OpenAI(base_url="http://localhost:11434")
model = outlines.from_vllm(openai_client)

result = model("Generate a fake social security number.", output_type, top_p=0.1)
print(result) # 782-32-3789
```

### Context-Free Grammar

SGLang supports grammars, but expects an EBNF format instead of the Lark format Outlines uses. Thus, to use a context-free grammar with SGLang, provide a string using the EBNF syntax to the Outlines `CFG` object.

```python
import openai
import outlines
from outlines.text import CFG

ebnf_grammar = """
root ::= answer
answer ::= "yes" | "no"
"""
output_type = CFG(ebnf_grammar)

openai_client = openai.OpenAI(base_url="http://localhost:11434")
model = outlines.from_vllm(openai_client)

result = sync_model("Is the weather good today?", output_type, extra_body={"guided_decoding_backend": "outlines"})
print(result) # 'yes'
```

## Inference Parameters

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `openai` client.

An optional parameter of particular interest is `extra_body`, which is a dictionnary containing arguments that are specific to vLLM and are not part of the standard `openai` interface.

See the [SGLang documentation](https://docs.sglang.ai/backend/openai_api_completions.html) on parameters for the OpenAI-compatible server for more information on inference parameters.
