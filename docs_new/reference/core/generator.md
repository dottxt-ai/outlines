---
title: Generator API
---

# Generator


The `Generator` class is the core component of Outlines v1. `Generator` accepts a [model](../reference/available_models) and an optional [output type](/api/types). If no output type is provided, the `Generator` will return unstructured text.

!!! note

    `Generator` is new as of Outlines v1, and replaces previous generator constructors:

    - `generate.cfg`
    - `generate.choice`
    - `generate.format`
    - `generate.fsm`
    - `generate.json`
    - `generate.regex`
    - `generate.text`

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
