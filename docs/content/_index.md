---
date: '2025-04-07T11:42:31-07:00'
title: ''
---

# Outlines

Outlines is a Python library maintained by [.TXT](https://dottxt.co) that guarantees structured output from 
large language models. It ensures LLMs speak the language of your 
application by making them follow specific formats such as JSON, regular expressions, or context-free grammars.

## Why use Outlines?

- It's fast
- Reliable LLM output
- Best-in-class JSON schema feature support
- Works with most popular inference backends like Transformers, vLLM, llama.cpp, etc.
- Use any open-weight model

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