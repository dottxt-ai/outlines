---
title: Dottxt
---

# Dottxt

!!! Installation

    To be able to use Dottxt in Outlines, you must install the `dottxt` python sdk: `pip install dottxt`

    You also need to have a Dottxt API key. This API key must either be set as an environment variable called `DOTTXT_API_KEY` or be provided to the `dottxt.client.Dottxt` class when instantiating it.

## Model Instantiation

To create an Anthropic model instance, you can use the `from_dottxt` function. It takes 2 arguments:
- `client`: an `dottxt.client.Dottxt` instance
- `model_name`: the name of the model you want to use (optional)
- `model_revision`: the name of the revision to use for the model selected (optional)

For instance:

```python
from dottxt.client import Dottxt
import outlines

# Create client
Dottxt(api_key="...")

# Create the model
model = outlines.from_dottxt(
    client,
    "meta-llama/Llama-3.1-8B",
    "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)
```

Use the `list_models` method of the Dottxt client to get a list of available model names and revisions for your account.

## Text Generation

Dottxt only supports constrained generation with structured objects, so output types that are equivalent to a JSON schema. You must always provide an output type as unconstrained generation is not available.

For instance:

```python
from typing import List
from dottxt.client import Dottxt
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_dottxt(
    Dottxt(),
    "meta-llama/Llama-3.1-8B",
    "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)

# Generate structured text
result = model("Create a character", Character)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
```

## Inference arguments

You can provide the same optional parameters you would pass to the `dottxt` sdk's client both during the initialization of the `Dottxt` class and when generating text.
Consult the [dottxt python sdk Github repository](https://github.com/dottxt-ai/dottxt-python) for the full list of parameters.
