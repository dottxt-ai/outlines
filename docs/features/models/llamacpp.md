---
title: llama.cpp
---

# llama.cpp

Outlines provides an integration with [Llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python library](https://github.com/abetlen/llama-cpp-python). Llamacpp allows to run quantized models on machines with limited compute.

!!! Installation

    You need to install the `llama-cpp-python` library to use the llama.cpp integration. Install all optional dependencies of the `LlamaCpp` model with: `pip install outlines[llamacpp]`.

    See the [llama-cpp-python Github page](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends) for instructions on installing with CUDA, Metal, ROCm and other backends.


## Model Initialization

To load the model, you can use the `from_llamacpp` function. The single argument of the function is a `Llama` model instance from the `llama_cpp` library. Consult the [Llama class API reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama) for detailed information on how to create a model instance and on the various available parameters.

For instance:

```python
import outlines
from llama_cpp import Llama

model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)
```

## Text Generation

To generate text, you can simply call the model with a prompt.

For instance:

```python
import outlines
from llama_cpp import Llama

# Create the model
model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)

# Call it to generate text
result = model("What's the capital of Latvia?", max_tokens=20)
print(result) # 'Riga'
```

#### Chat

You can also use chat inputs with the `LlamaCpp` model. To do so, call the model with a `Chat` instance.

For instance:

```python
import outlines
from llama_cpp import Llama
from outlines.inputs import Chat

# Create the model
model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)

# Create the prompt containing the text and the image
prompt = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "assistant", "content": "What's the capital of Latvia?"},
])

# Call the model to generate a response
response = model(prompt, max_tokens=50)
print(response) # 'Riga.'
```

#### Streaming

The `LlamaCpp` model also supports streaming.

For instance:

```python
import outlines
from llama_cpp import Llama

# Create the model
model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)

# Stream text
for chunk in model.stream("Write a short story about a cat.", max_tokens=100):
    print(chunk) # 'In...'
```

## Structured Generation

The `LlamaCpp` model supports all output types available in Outlines. Simply provide an `output_type` after the prompt when calling the model.

### Basic Type

```python
import outlines
from llama_cpp import Llama

output_type = int

model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)

result = model("How many countries are there in the world?", output_type)
print(result) # '200'
```

### JSON Schema

```python
from typing import List
from pydantic import BaseModel
import outlines
from llama_cpp import Llama

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)

result = model("Create a character.", output_type=Character, max_tokens=200)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

### Multiple Choice

```python
from typing import Literal
import outlines
from llama_cpp import Llama

output_type = Literal["Paris", "London", "Rome", "Berlin"]

model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)

result = model("What is the capital of France?", output_type)
print(result) # 'Paris'
```

### Regex

```python
from outlines.types import Regex
import outlines
from llama_cpp import Llama

output_type = Regex(r"\d{3}-\d{2}-\d{4}")

model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)

result = model("Generate a fake social security number.", output_type)
print(result) # '782-32-3789'
```

### Context-free grammar

```python
from outlines.types import CFG
import outlines
from llama_cpp import Llama

output_type = CFG("""
root ::= answer
answer ::= "yes" | "no"
""")

model = outlines.from_llamacpp(
    Llama.from_pretrained(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    )
)

result = model("Are you feeling good today?", output_type)
print(result) # 'yes'
```

## Inference Arguments

When calling the model, you can provide optional inference parameters on top of the prompt and the output type. These parameters will be passed on to the `__call__` method of the `llama_cpp.Llama` model. Some common inference arguments include `max_tokens`, `temperature`, `frequency_penalty` and `top_p`.

See the [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__) for more information on inference parameters.
