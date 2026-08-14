---
title: Dottxt
---

# Dottxt

!!! Installation

    You need to install the `dottxt` python sdk to be able to use the Dottxt API in Outlines. Install all optional dependencies of the `Dottxt` model with: `pip install "outlines[dottxt]"`.

    You also need to have a Dottxt API key. Request one [here](https://h1xbpbfsf0w.typeform.com/to/fwQNWmS8?utm_source=github&utm_medium=organic&utm_campaign=outlines). Once obtained, the key must either be set as an environment variable called `DOTTXT_API_KEY` or be provided to the `dottxt.client.DotTxt` class when instantiating it.

## Model Initialization

### Synchronous

Use `from_dottxt` to create a model instance from a `dottxt.DotTxt` client. The `model` argument is the identifier of the model to use. It can be set at initialization time or passed per call via the `model=` keyword argument — but it must be provided somewhere or a `ValueError` is raised.

```python
from dottxt.client import DotTxt
import outlines

client = DotTxt(api_key="...")
model = outlines.from_dottxt(client, "dottxt/dottxt-v1-alpha")
```

Use `client.models.list()` to get the list of available model identifiers for your account.

### Asynchronous

Pass a `dottxt.AsyncDotTxt` client to the same `from_dottxt` function — it detects the client type and returns an `AsyncDottxt` instance:

```python
from dottxt.client import AsyncDotTxt
import outlines

client = AsyncDotTxt(api_key="...")
model = outlines.from_dottxt(client, "dottxt/dottxt-v1-alpha")
```

## Text Generation

Dottxt only supports constrained generation with JSON schema output types. You must always provide a value for the `output_type` parameter as unconstrained generation is not available.

### Synchronous

```python
from typing import List
from pydantic import BaseModel
from dottxt.client import DotTxt
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

model = outlines.from_dottxt(DotTxt(), "dottxt/dottxt-v1-alpha")

result = model("Create a character", Character)
print(result)  # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result))  # name=Evelyn, age=34, skills=[...]
```

### Asynchronous

```python
from typing import List
from pydantic import BaseModel
from dottxt.client import AsyncDotTxt
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

model = outlines.from_dottxt(AsyncDotTxt(), "dottxt/dottxt-v1-alpha")

result = await model("Create a character", Character)
print(result)  # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
```

## Inference arguments

Optional parameters supported by the `dottxt` SDK's `generate` method can be passed at generation time as keyword arguments: `temperature`, `max_tokens`, `seed`, or any additional OpenAI-compatible chat completion parameter.

```python
result = model("Create a character", Character, temperature=0.8, max_tokens=256)
```

The `model` identifier can also be overridden per call this way:

```python
result = model("Create a character", Character, model="dottxt/other-model")
```

Consult the [dottxt python sdk GitHub repository](https://github.com/dottxt-ai/dottxt-python) for the full list of parameters.
