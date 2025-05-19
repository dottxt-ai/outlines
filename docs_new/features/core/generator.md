---
title: Generator
---

# Generator

The `Generator` constructor is the core component of Outlines v1. It accepts a [model](../reference/available_models) and an optional [output type](/api/types), and returns a callable `generator` object that can be used to generate text from a prompt.

!!! note

    `Generator` is new as of Outlines v1. It replaces previous generator constructors that are now deprecated:

    - `generate.cfg`
    - `generate.choice`
    - `generate.format`
    - `generate.fsm`
    - `generate.json`
    - `generate.regex`
    - `generate.text`

## Parameters

- **model**: An instance of an Outlines [model](../reference/available_models)
  - If the model provided is aysnc, the constructor will return an async generator

- **output_type**: Optional. The desired output structure expressed as an [output type](/api/types)
  - If `None`, the generator will generate unconstrained text

- **processor**: Optional, keyword-only. An instance of an `OutlinesLogitsProcessor` (see documentation on [logits processors](/api/logits_processors))
  - Advanced usage for directly providing a custom logits processor.
  - Only supported for `SteerableModel` instances.

 The `output_type` and `processor` parameters are mutually exclusive. Provide at most one.

## Returns

The constructor returns a `generator` object that contains the methods below. If you provide an async model, you will get an async generator.

### `__call__(prompt, **inference_kwargs)`

Generate text for the given prompt with the model given to the `Generator`. The response will follow the structure provided through the output type.

- **prompt**: The input text prompt.
- **inference_kwargs**: Additional keyword arguments passed to the model's text generation function.
- **Returns**: The generated text conforming to the specified output type.

### `stream(prompt, **inference_kwargs)`

Generate text generation for the given prompt with the model given to the `Generator`. The response will follow the structure provided through the output type.

- **prompt**: The input text prompt.
- **inference_kwargs**: Additional keyword arguments passed to the model's text generation function.
- **Returns**: An iterator (or async iterator for async models) that yields text chunks as they're generated.

#### Text generation return type

The `__call__` and `stream` methods described above always return a raw strings (or a list of strings in case of batch generation), even when using constrained generation. You must cast the string into the desired format. For instance, if you provided a Pydantic model as output type, you should use the `model_validate_json` of the model to parse the generation response as a Pydantic model instance.

## Usage

### Basic Usage

```python
from outlines import Generator, from_transformers
import transformers

# Create a model
model = from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    transformers.AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

# Create a generator for unconstrained text and call it to generate text
generator = Generator(model)
response = generator("Tell me about machine learning.", max_new_tokens=20)

print(result) # "Machine learning is a subset of artificial intelligence (AI) that provides..."
```

### Generating Structured Output

```python
from typing import List
from pydantic import BaseModel
from outlines import Generator, from_llamacpp
from llama_cpp import Llama

# Create a model
model = from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/phi-2-GGUF",
        filename="phi-2.Q4_K_M.gguf",
    )
)

# Define the output structure
class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create a generator for the output type defined and call it to generate text
generator = Generator(model, Character)
response = generator("Create a character for my game", max_tokens=100)

print(response) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'

# Cast into Pydantic model instance
response = Character.model_validate_json(response)
print(response) # name='Evelyn' age=34 skills=['archery', 'stealth', 'alchemy']
```

### Streaming Generation

```python
from outlines import Generator, from_openai
import openai

# Create a model
model = from_openai(
    openai.OpenAI(),
    "gpt-4o-mini",
)

# Create a generator
generator = Generator(model)

# Stream the output chunks
for chunk in generator.stream("Generate a response..."):
    print(chunk)
```

### Async Models

```python
import asyncio
from outlines import Generator, from_tgi
from huggingface_hub import AsyncInferenceClient

async def generate():
    model = from_tgi(
        AsyncInferenceClient("http://localhost:8000/v1")
    )
    generator = Generator(model)

    # Async generation
    response = await generator("Summarize this article.")

    # Async streaming
    async for chunk in generator.stream("Generate step by step..."):
        print(chunk)

asyncio.run(generate())
```

## Notes

- Not all models support all output types. Consult the documentation of the [model](../reference/models) you want to use to check the output types available
- Some models do not support streaming
- The generator's call/stream methods accept inference keyword arguments that are passed on to underlying the inference model Outlines wraps, such as Transformers or vLLM. Review the documentation of the inference library in question to find what arguments to use.
- The generator will always return strings. You need to cast the response to your
