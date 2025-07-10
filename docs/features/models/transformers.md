---
title: Transformers
---

# Transformers

!!! Installation

    You need to install the `transformers` library to be able to use the Transformers in Outlines. Install all optional dependencies of the `Transformers` model with: `pip install outlines[transformers]`.

    See the [HuggingFace documentation](https://huggingface.co/docs/transformers/en/installation) for more information on installing `transformers` with CPU, GPU...

## Model Initialization

To load the model, you can use the `from_transformers` function. It takes 2 arguments:

- `model`: a `transformers` model (created with `AutoModelForCausalLM` for instance)
- `tokenizer_or_processor`: a `transformers` tokenizer (created with `AutoTokenizer` for instance, it must be an instance of either `PreTrainedTokenizer` or `PreTrainedTokenizerFast`)

For instance:

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create the transformers model and tokenizer
hf_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create the Outlines model
model = outlines.from_transformers(hf_model, hf_tokenizer)
```

If you provide a processor instead of a tokenizer for the second argument of the `from_transformers` function, you would get a `TransformersMultiModal` instance. See the [TransformersMultiModal model documentation](./transformers_multimodal.md) for more information on using multimodal models in Outlines.

## Text Generation

To generate text, you can simply call the model with a prompt.

For instance:

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create model
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

# Call it to generate text
result = model("What's the capital of Latvia?", max_new_tokens=20)
print(result) # 'Riga'
```

#### Chat

You can also use chat inputs with the `Transformers` model. To do so, call the model with a `Chat` instance. The content of messsage within the chat can be vision inputs as described above.

For instance:

```python
import outlines
from outlines.inputs import Chat
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create the model
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

# Create the chat input
prompt = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of Latvia?"},
])

# Call the model to generate a response
response = model(prompt, max_new_tokens=50)
print(response) # 'This is a picture of a black dog.'
```

#### Batching

Finally, the `Transformers` model supports batching through the `batch` method. To use it, provide a list of prompts (using the formats described above) to the `batch` method. You will receive as a result a list of completions.

For instance:

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create model
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
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

As a local model, `Transformers` supports all output types available in Outlines. Simply provide an `output_type` after the prompt when calling the model.

### Simple Type

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

output_type = int

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

result = model("How many countries are there in the world?", output_type, max_new_tokens=5)
print(result) # '200'
```

### JSON Schema

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from typing import List

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

result = model("Create a character.", output_type=Character, max_new_tokens=200, repetition_penalty=0.5)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

### Multiple Choice

```python
from typing import Literal
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

output_type = Literal["Paris", "London", "Rome", "Berlin"]

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

result = model("What is the capital of France?", output_type, max_new_tokens=10, temperature=0)
print(result) # 'Paris'
```

### Regex

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines.types import Regex

output_type = Regex(r"\d{3}-\d{2}-\d{4}")

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

result = model("Generate a fake social security number.", output_type, max_new_tokens=20, top_p=0.5)
print(result) # '782-32-3789'
```

### Context-Free Grammar

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
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

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

result = model("Write an addition.", output_type, max_new_tokens=100)
print(result) # '23 + 48'
```

## Inference Arguments

When calling the model, you can provide optional inference parameters on top of the prompt and the output type. These parameters will be passed on to the `generate` method of the `transformers` model. Some common inference arguments include `max_new_tokens`, `temperature`, `repetition_penalty` and `top_p`.

See the [transformers documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation) for more information on inference parameters.

!!! Warning

    The `max_new_tokens` inference parameter has a default value of 20. This is insufficient for most tasks and will result in the generation output not respecting the output type (because the response is truncated). We recommend you always provide a value for this argument.
