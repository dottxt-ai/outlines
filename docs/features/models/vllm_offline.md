---
title: vLLM Offline
---

# vLLM Offline

Outlines provides an integration with [vLLM](https://docs.vllm.ai/en/latest/) using the [vllm library](https://github.com/vllm-project/vllm). This model allows you to use vLLM in the "Offline Inference" mode, meaning that text generation happens within the model, there is no separate server. If you want to use vLLM with a server, see the [VLLM model documentation](./vllm.md).

!!! Note "Installation"

    You need to install the `vllm` library to be able to use the `VLLMOffline` model: `pip install vllm`. Due to a library version conflict between outlines and vllm, you MUST install `vllm` before installing `outlines`.

    When installing `outlines` (after having first installed `vllm`), you may encounter the following error: `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed`. You can safely ignore it.

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

!!! Note

    When initializing the `vllm.LLM` object, you can specify a `guided_decoding_backend` to choose what library will be used by vLLM to constrain the generation. Consult the [vLLM documentation](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html) on structured output for the list of possible values.

## Text Generation

Once you've created your Outlines `VLLMOffline` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt.

For instance:

```python
import outlines
from vllm import LLM

# Create the model
model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

# Call it to generate text
response = model("What's the capital of Latvia?", max_tokens=20)
print(response) # 'Riga'
```

#### Chat

You can also use chat inputs with the `VLLMOffline` model. To do so, call the model with a `Chat` instance. The content of messsage within the chat can be vision inputs as described above.

For instance:

```python
import outlines
from vllm import LLM
from outlines.inputs import Chat

# Create the model
model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

# Create the chat prompt
prompt = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of Latvia?"},
])

# Call the model to generate a response
response = model(prompt, max_tokens=50)
print(response) # 'Riga'
```

#### Streaming

The `VLLMOffline` model supports streaming through the `stream` method.

For instance:

```python
import outlines
from vllm import LLM

# Create the model
model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

# Stream the response
for chunk in model.stream("Tell me a short story about a cat.", max_tokens=50):
    print(chunk) # 'Once...'
```

#### Batching

Finally, the `VLLMOffline` model also supports batching through the `batch` method. To use it, provide a list of prompts (using the formats described above) to the `batch` method. You will receive as a result a list of completions.

For instance:

```python
import outlines
from vllm import LLM

# Create the model
model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

# Create a list of prompts that will be used in a single batch
prompts = [
    "What's the capital of Lithuania?",
    "What's the capital of Latvia?",
    "What's the capital of Estonia?"
]

# Call it to generate text
result = model.batch(prompts, max_new_tokens=20)
print(result) # ['Vilnius', 'Riga', 'Tallinn']
```

## Structured Generation

The `VLLMOffline` model supports all output types available in Outlines. Simply provide an `output_type` after the prompt when calling the model.

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
from vllm import LLM, SamplingParams
from typing import List
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

result = model("Create a character.", output_type=Character, sampling_params=SamplingParams(frequency_penalty=1.5, max_tokens=200))
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

### Multiple Choice

```python
from typing import Literal
import outlines
from vllm import LLM, SamplingParams

output_type = Literal["Paris", "London", "Rome", "Berlin"]

model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

result = model("What is the capital of France?", output_type, sampling_params=SamplingParams(temperature=0))
print(result) # 'Paris'
```

### Regex

```python
import outlines
from vllm import LLM, SamplingParams
from outlines.types import Regex

output_type = Regex(r"\d{3}-\d{2}-\d{4}")

model = outlines.from_vllm_offline(
    LLM("microsoft/Phi-3-mini-4k-instruct")
)

result = model("Generate a fake social security number.", output_type, sampling_params=SamplingParams(top_p=0.1))
print(result) # '782-32-3789'
```

### Context-Free Grammar

```python
import outlines
from vllm import LLM, SamplingParams
from outlines.types import CFG

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

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `generate` method of the `LLM` model instance. An argument of particular interest is `sampling_params`. It takes as a value a `vllm.SamplingParams` instance containing parameters such as max_tokens or temperature.

See the [vLLM documentation](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html#vllm.sampling_params.SamplingParams) on sampling parameters for more information on inference parameters.
