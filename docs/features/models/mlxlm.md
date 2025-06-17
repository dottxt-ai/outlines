---
title: mlx-lm
---

# mlx-lm

Outlines provides an integration with [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms), allowing models to be run quickly on Apple Silicon via the [mlx](https://ml-explore.github.io/mlx/build/html/index.html) library.

!!! Note "Installation"

    You need a device that [supports Metal](https://support.apple.com/en-us/102894) to use the mlx-lm integration.

    You need to install the `mlx` and `mlx-lm` libraries to be able to use mlx in Outlines. Install all optional dependencies of the `MLXLM` model with: `pip install outlines[mlxlm]`.

## Model Initialization

To create a MLXLM model instance, you can use the `from_mlxlm` function. It takes 2 arguments:

- `model`: an `mlx.nn.Module` instance
- `tokenizer`: a `transformers.PreTrainedTokenizer` instance

However, we recommend you simply pass on the output of the `mlx_lm.load` function (it takes a model name as an argument).

For instance:

```python
import outlines
import mlx_lm

# Create the model
model = outlines.from_mlxlm(
    *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
)
```

## Text Generation

To generate text, you can simply call the model with a prompt.

For instance:

```python
import outlines
import mlx_lm

# Load the model
model = outlines.from_mlxlm(
    *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
)

# Call it to generate text
result = model("What's the capital of Latvia?", max_tokens=20)
print(result) # 'Riga'
```

The `MLXLM` model also supports streaming. For instance:

```python
import outlines
import mlx_lm

# Load the model
model = outlines.from_mlxlm(
    *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
)

# Stream text
for chunk in model.stream("Write a short story about a cat.", max_tokens=100):
    print(chunk) # 'In...'
```

## Structured Generation

As a local model, `MLXLM` supports all forms of structured generation available in Outlines.

### Basic Type

```python
import outlines
import mlx_lm

output_type = int

model = outlines.from_mlxlm(
    *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
)

result = model("How many countries are there in the world?", output_type)
print(result) # '200'
```

### JSON Schema

```python
from pydantic import BaseModel
from typing import List
import outlines
import mlx_lm

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

model = outlines.from_mlxlm(
    *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
)

result = model("Create a character.", output_type=Character)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

### Multiple Choice

```python
from typing import Literal
import outlines
import mlx_lm

output_type = Literal["Paris", "London", "Rome", "Berlin"]

model = outlines.from_mlxlm(
    *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
)

result = model("What is the capital of France?", output_type)
print(result) # 'Paris'
```

### Regex

```python
from outlines.types import Regex
import outlines
import mlx_lm

output_type = Regex(r"\d{3}-\d{2}-\d{4}")

model = outlines.from_mlxlm(
    *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
)

result = model("Generate a fake social security number.", output_type)
print(result) # '782-32-3789'
```

### Context-Free Grammar

```python
from outlines.types import CFG
import outlines
import mlx_lm

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

model = outlines.from_mlxlm(
    *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
)

result = model("Write an addition.", output_type, max_tokens=20)
print(result) # '23 + 48'
```

## Inference Arguments

When calling the model, you can provide optional inference parameters on top of the prompt and the output type. These parameters will be passed on to the `mlx_lm.generate` function used to generate text.

See the [MLXLM documentation](https://github.com/ml-explore/mlx-lm) for more information on inference parameters.
