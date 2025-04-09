---
date: '2025-04-07T11:42:31-07:00'
title: ''
---

# Outlines

Outlines is a Python library from [.TXT](https://dottxt.co) that guarantees structured output from 
large language models. It ensures LLMs speak the language of your application by making them follow specific formats such as JSON, regular expressions, or context-free grammars.

> [!NOTE]
> We need a "get started" button here that's super obvious

{{< cards >}}
  {{< card title="Get started" link="/docs/getting_started" >}}
{{< /cards >}}

## Features

> [!NOTE]
> These cards should have action/callout links to go places

{{< cards >}}
  {{< card title="Reliable" subtitle="Guaranteed schema compliance -- always valid JSON" icon="shield-check" >}}
  {{< card title="Feature-rich" subtitle="Supports a large proportion of the JSON Schema spec, along with regex and context-free grammars" icon="puzzle" >}}
  {{< card title="Fast" subtitle="Outlines has negligible runtime overhead, and fast compilation times" icon="lightning-bolt" >}}
  {{< card title="Universal" subtitle="Outlines is a powered by Rust, and can be easily bound to other languages" icon="cog" >}}
  {{< card title="Simple" subtitle="Outlines is a low-abstraction library. Write code the way you normally do with LLMs. No agent frameworks needed." icon="light-bulb" >}}
  {{< card title="Powerful" subtitle="Manage prompt complexity with prompt templating" icon="search" >}}
{{< /cards >}}

## Installation

We recommend using `uv` to install Outlines. You can find installation instructions [here](https://github.com/astral-sh/uv).

```bash
uv pip install 'outlines[transformers]'
```

or the classic `pip`:

```bash
pip install 'outlines[transformers]'
```

## Quick start

> [!NOTE]
> - [ ] This section should include tabs for each of the different inference backends.
> - [ ] Add comments to the code example to explain what it does.

### JSON

```python
import json
from outlines import models
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

class Person(BaseModel):
    name: str
    age: int
    email: str = Field(pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = models.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_id),
    AutoTokenizer.from_pretrained(model_id)
)

person_text = """
John Doe
30
john.doe@example.com
"""

result = model(
    f"Extract the person information from this text:\n{person_text}", 
    Person,
    max_new_tokens=100
)
print(result)
```

Result:
```json
{
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com"
}
```

### Regex

> [!NOTE]
> Insert regex example here

### Multiple choice

> [!NOTE]
> Insert multiple choice example here

## Supported models

> [!NOTE]
> Provide full model list with links to docs about each model

- vLLM
- Transformers
- OpenAI


## About .txt

Outlines is built with ❤️ by [.txt](https://dottxt.co). 

.txt solves the critical problem of reliable structured output generation for large language models. Our commercially-licensed libraries ensure 100% compliance with JSON Schema, regular expressions and context-free grammars while adding only microseconds of latency. Unlike open-source alternatives, we offer superior reliability, performance, and enterprise support.

Schedule a [demo call](https://cal.com/team/dottxt/sales) to learn more about how .txt can help you integrate LLMs into production environments without additional engineering resources.
