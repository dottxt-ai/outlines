---
title: Structured Generation Backends
---

# Structured Generation Backends

Outlines relies on a structured generation backend to control text generation for steerable models such thah they conform to the output type provided. One of those backends is of course `outlines-core`, but you also have access to two other libraries that fulfill the same purpose: `llguidance` and `xgrammar`.

## Overview

To select the backend to use for your generation, provide a value for the `backend` argument when calling a model or a generator.

For instance:

```python
from typing import Literal
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

output_type = Literal["Paris", "London", "Rome", "Berlin"]

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

result = model("What is the capital of France?", output_type, backend="llguidance")
print(result) # 'Paris'

generator = outlines.Generaor(model, output_type)
result = generator("What is the capital of France?", backend="xgrammar")
print(result) # 'Paris'
```

If you do not provide a value for the `backend` argument, the default value will be used. The default value depends on the type of output type:

- JSON schema: `outlines_core`
- Regex: `outlines_core`
- Context-free grammar: `llguidance`

## Features matrix

As mentioned previously, selecting the structured generation backend is only applicable to steerable models, so `Transformers`, `LlmaCpp` and `MLXLM`. Additionaly, some backends do not support some models within those or some output types.

| | outlines_core | llguidance | xgrammar |
|---|---|---|---|
| **Models** | | | |
| Transformers | ✅ | ✅ | ✅ |
| LlamaCpp | ✅ | ✅ | ❌ |
| MLXLM | ✅ | ✅ | ✅ |
| **Output Types** | | | |
| JSON Schema | ✅ | ✅ | ✅ |
| Regex | ✅ | ✅ | ✅ |
| Grammar | ❌ | ✅ | ✅ |
