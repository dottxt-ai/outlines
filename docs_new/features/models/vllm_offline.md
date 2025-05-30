---
title: vLLM Offline
---

# vLLM Offline

Outlines provides an integration with [vLLM](https://docs.vllm.ai/en/latest/) using the [vllm library](https://github.com/vllm-project/vllm). This model allows you to use vLLM in the "Offline Inference" mode, meaning that text generation happens within the model, there is no separate server. If you want to use vLLM with a server, see the [VLLM model documentation](./vllm.md).

!!! Note "Installation"

    You need to install the `vllm` library to be able to use the `VLLMOffline` model: `pip install vllm`.

    See the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) for instructions on how to install vLLM for CPU, ROCm...

## Model Initialization

To load the model, you can use the `from_vllm_offline` function. The single argument of the function is a `LLM` model instance from the `vllm` library. You will then receive a `VLLMOffline` model instance you can use to generate text.

Consult the [LLM class API reference](https://docs.vllm.ai/en/latest/api/vllm/index.html#vllm.LLM) for detailed information on how to create an `LLM` instance and on the various available parameters.

For instance:

```python
import outlines
from vllm import LLM

# Create the model
model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)
```

## Text Generation

To generate text, you can simply call the model with a prompt.

For instance:

```python
import outlines
from vllm import LLM

# Create the model
model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

# Call it to generate text
result = model("What's the capital of Latvia?", max_tokens=20)
print(result) # 'Riga'
```

## Structured Generation

As a local model, `VLLMOffline` supports all output types available in Outlines. Simply provide an `output_type` after the prompt when calling the model.

### Simple Type

```python
import outlines
from vllm import LLM

output_type = int

model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

result = model("How many countries are there in the world?", output_type)
print(result) # '200'
```

### JSON Schema

```python
import outlines
from vllm import LLM
from typing import List
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

result = model("Create a character.", output_type=Character, frequency_penalty=1.5)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

### Multiple Choice

```python
from typing import Literal
import outlines
from vllm import LLM

output_type = Literal["Paris", "London", "Rome", "Berlin"]

model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

result = model("What is the capital of France?", output_type, temperature=0)
print(result) # 'Paris'
```

### Regex

```python
import outlines
from vllm import LLM
from outlines.types import Regex

output_type = Regex(r"\d{3}-\d{2}-\d{4}")

model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

result = model("Generate a fake social security number.", output_type, top_p=0.1)
print(result) # '782-32-3789'
```

### Context-Free Grammar

```python
import outlines
from vllm import LLM
from outlines.text import CFG

arithmetic_grammar = """
?start: sum

?sum: product
| sum "+" product   -> add
| sum "-" product   -> sub

?product: atom
| product "*" atom  -> mul
| product "/" atom  -> div

?atom: NUMBER           -> number
| "-" atom         -> neg
| "(" sum ")"

%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""
output_type = CFG(arithmetic_grammar)

model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

result = model("Write an addition.", output_type)
print(result) # '23 + 48'
```

## Inference Arguments

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `generate` method of the `LLM` model instance. Some common inference arguments include `max_tokens`, `temperature`, `frequency_penalty` and `top_p`.

See the [vLLM documentation](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html#vllm.sampling_params.SamplingParams) on sampling parameters for more information on inference parameters.
