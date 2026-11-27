---
title: Generator API
---

# Generator


The `Generator` class is the core component of Outlines v1. `Generator` accepts a [model](../models/index.md) and an optional [output type](../core/output_types.md). If no output type is provided, the `Generator` will return unstructured text.

!!! note

    `Generator` is new as of Outlines v1, and replaces previous generator constructors:

    - `generate.cfg`
    - `generate.choice`
    - `generate.format`
    - `generate.fsm`
    - `generate.json`
    - `generate.regex`
    - `generate.text`

## Methods

Generators implement the same methods as models:

- `__call__`
- `batch`
- `stream`

All of them take a single positional argument: the [model input](../core/inputs.md) from which text is generated. Contrarily to the equivalent methods of models, you do not need to provide an output type as it has already been defined when initializing the generator.

## Basic Usage

```python
from outlines import Generator, from_transformers
import transformers

# Initialize a model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(model_name),
    transformers.AutoTokenizer.from_pretrained(model_name),
)

# Create a generator for plain text
generator = Generator(model)
result = generator("Write a short poem about AI.")

# Print the result
print(result)
```

## Structured Generation

```python
from pydantic import BaseModel
from outlines import Generator, from_transformers
import transformers

# Define a Pydantic model for structured output
class BookRecommendation(BaseModel):
    title: str
    author: str
    year: int

# Initialize a model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(model_name),
    transformers.AutoTokenizer.from_pretrained(model_name),
)

# Create a generator for JSON output
generator = Generator(model, BookRecommendation)

# Generate a book recommendation
result = generator("Recommend a science fiction book.")

# Parse the JSON result into a Pydantic model
book = BookRecommendation.model_validate_json(result)
print(f"{book.title} by {book.author} ({book.year})")
```

## Optional Memory Augmentation

`MemoryAugmentedGenerator` wraps an existing generator to recall previous outputs
as examples and store each completed output. Install and configure your memory
client separately; Outlines does not require a memory SDK. For example, with a
synchronous `DakeraClient` and the model and schema from the example above:

```python
import os

from dakera import DakeraClient
from outlines import Generator, MemoryAugmentedGenerator

memory = DakeraClient(
    base_url="http://localhost:3300",
    api_key=os.environ["DAKERA_API_KEY"],
)
generator = MemoryAugmentedGenerator(
    Generator(model, BookRecommendation),
    memory,
    agent_id="book-recommendations",
    top_k=2,
)
result = generator("Recommend a science fiction book.", max_new_tokens=200)
book = BookRecommendation.model_validate_json(result)
```

The memory client must implement synchronous `recall(agent_id=..., query=...,
top_k=...)` and `store_memory(agent_id=..., content=..., metadata=...)` methods.
Recall may return `None` or an object with a `memories` sequence whose entries
have a `content` string. `agent_id` selects the memory namespace and defaults to
`"outlines-agent"`; `top_k` must be a positive integer and defaults to `2`.

Each call queries memory using the original prompt. Recalled outputs are
prepended as examples; when recall returns no memories, the original prompt is
passed unchanged. Generation options are forwarded to the wrapped generator,
which continues to enforce its output constraints. The raw generated string is
stored with `metadata={"source": prompt[:100]}` and returned unchanged. Memory
quality and filtering are the memory backend's responsibility.

This wrapper supports synchronous, non-streaming calls with string prompts and
a single raw string result. Multiple outputs and other non-string results raise
`TypeError` without being stored. The constructor rejects asynchronous generators
and asynchronous memory operations. The wrapper does not provide batch,
streaming, or asynchronous methods. Recall, generation, and storage errors
propagate to the caller; a storage failure therefore raises instead of returning
the generated result.

## Parameters

- `model`: The language model to use for generation
- `output_type`: Optional. The type of output to generate

## Generation Parameters

When calling the generator, you can pass additional parameters to control the generation process. These parameters are passed through to the underlying model, so they depend on the specific model being used.

Common parameters for most models include:
- `max_new_tokens`: Maximum number of tokens to generate
- `temperature`: Controls randomness (higher values = more random)
- `top_p`: Controls diversity via nucleus sampling
- `stop_strings`: String or list of strings at which to stop generation

Example:
```python
result = generator(
    "Write a short story.",
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    stop_strings=["THE END", "###"]
)
```

## Return Value

The generator always returns a raw string containing the generated text. When generating structured outputs, you need to parse this string into the desired format.

Unlike in Outlines v0, where the return type could be a parsed object, in v1 you are responsible for parsing the output when needed:

```python
# Outlines v1 approach
from pydantic import BaseModel
from outlines import Generator

class Person(BaseModel):
    name: str
    age: int

generator = Generator(model, Person)
result = generator("Generate a person:")

# Parse the result yourself
person = Person.model_validate_json(result)
```

::: outlines.generator.Generator

::: outlines.memory.MemoryAugmentedGenerator
