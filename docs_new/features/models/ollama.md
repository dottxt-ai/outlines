---
title: Ollama
---

# Ollama

!!! Installation

    To be able to use Ollama in Outlines, you must install both Ollama and the `ollama` python sdk.

    - To download Ollama: https://ollama.com/download
    - To install the ollama python sdk: `pip install ollama`

    Consult the [`ollama` documentation](https://github.com/ollama/ollama-python) for detailed informations about how to initialize models and the available options.

## Model Initialization

To create an Ollama model instance, you can use the `from_ollama` function. It takes 2 arguments:
- `client`: an `ollema.Client` instance
- `model_name`: the name of the model you want to use

For instance:

```python
import ollama
import outlines

# Create the client
client = ollama.Client()

# Create the model
model = outlines.from_ollama(
    client,
    "tinyllama",
)
```

You can find the list of available models on the [Ollama library](https://ollama.com/library).

## Text Generation

Once you've created your Outlines `Ollama` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt. The method accepts all arguments that you could pass to the `generate` method of the Ollama client as keyword arguments. Ollama supports streaming

For instance:

```python
import ollama
import outlines

# Create the model
model = outlines.from_ollama(ollama.Client(), "tinyllama")

# Stream text
for chunk in model.stream("What's the capital of Latvia?"):
    print(result)
```

## Structured Generation

Ollama only supports structured generation based on a JSON schema.

For instance:

```python
from typing import List
import ollama
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_ollama(ollama.Client(), "tinyllama")

# Call it with the ouput type to generate structured text
result = model("Create a character", Character)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
```

## Optional parameters

You can provide the same optional parameters you would pass to the `ollama` sdk's client both during the initialization of the `Ollama` class and when generating text.
Consult the [Ollama REST API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) for the full list of parameters.
