---
date: '2025-04-07T11:48:38-07:00'
draft: true
title: 'Migration guide'
---

# Outlines 1.0 migration guide

Outlines 1.0 introduces some breaking changes that affect the way you use the library.

This guide will help you migrate your code to the new version.

All previous functionality will be supported until Outlines version 1.5.0.

## TODO: things to mention

### Removed

- Generate functions like `generate.json`, `generate.choice`, `generate.regex`, and `generate.cfg` have been removed.
- `load_lora` has been removed from the `VLLM` model. You must pass the `lora_request` argument as a keyword argument when calling the model or generator.
- `load_lora` has been completely removed from the LlamaCpp backend.


### Warnings to explain:

```
The `openai` function is deprecated starting from v1.0.0.
Do not use it. Support for it will be removed in v1.5.0.
Instead, you should instantiate a `OpenAI` model with the
`outlines.from_openai` function that takes an openai library
client and a model name as arguments. Similarly, you cannot
instantiate a `OpenAI` model directly with a `OpenAIConfig`
instance anymore, but must provide a client and a model name
instead.
For example:
```python
from openai import OpenAI as OpenAIClient
from outlines import from_openai
client = OpenAIClient()
model = from_openai(client, "gpt-4o")
```

```
The `load_lora` method is deprecated starting from v1.0.0.
Support for it will be removed in v1.5.0.
Please use the v1 of the `outlines` library by using the
`outlines.from_vllm` function to create a `VLLM` model
instance.
In the v1, you must pass the `lora_request` argument as
a keyword argument when calling the model or generator.
```





