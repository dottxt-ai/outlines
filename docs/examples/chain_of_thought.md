# Chain of thought


Chain of thought is a prompting technique introduced in the paper ["Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903) where throught prompting the authors generate a series of intermediate reasoning steps which improves the ability of LLMs to perform complex reasoning.

In this guide, we use [outlines](https://dottxt-ai.github.io/outlines/) to apply chain of thought through structured output.

We use [llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) library. Outlines supports llama-cpp-python, but we need to install it ourselves:

```shell
pip install llama-cpp-python
```

To create an outlines `LlamaCpp` model, you first need to create a `Llama` object from the `llama-cpp-python` library. Then you can create the outlines model by calling `models.from_llamacpp` with the `Llama` object instance as argument. To create the `Llama` object, you need to provide the model weights by passing the name of the repository on the HuggingFace Hub, and the filenames or glob pattern (it will automatically download the weights from the hub):

```python
import llama_cpp
import outlines

llm = llama_cpp.Llama(
    "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
        "NousResearch/Hermes-2-Pro-Llama-3-8B"
    ),
    n_gpu_layers=-1,
    flash_attn=True,
    n_ctx=8192,
    verbose=False
)
model = outlines.from_llamacpp(llm)
```

??? note "(Optional) Store the model weights in a custom folder"

    By default the model weights are downloaded to the hub cache but if we want so store the weights in a custom folder, we pull a quantized GGUF model [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF) by [NousResearch](https://nousresearch.com/) from [HuggingFace](https://huggingface.co/):

    ```shell
    wget https://hf.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf
    ```

    We initialize the model:

    ```python
    from llama_cpp import Llama

    llm = Llama("/path/to/model/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf", ...)
    ```

## Chain of thought

We first define our Pydantic class for a reasoning step:

```python
from pydantic import BaseModel, Field

class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="Reasoning step")
```

We then define the Pydantic class for reasoning which will consist on a list of reasoning steps and a conclusion, and we get its JSON schema:

```python
from typing import List

class Reasoning(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: str = Field(..., description="Conclusion")

json_schema = Reasoning.model_json_schema()
```

We then need to adapt our prompt to the [Hermes prompt format for JSON schema](https://github.com/NousResearch/Hermes-Function-Calling?tab=readme-ov-file#prompt-format-for-json-mode--structured-outputs):

```python
from outlines import Template

generate_hermes_prompt = Template.from_string(
    """
    <|im_start|>system
    You are a world class AI model who answers questions in JSON
    Here's the json schema you must adhere to:
    <schema>
    {{ json_schema }}
    </schema>
    <|im_end|>
    <|im_start|>user
    {{ user_prompt }}
    <|im_end|>
    <|im_start|>assistant
    <schema>
    """
)
```

For a given user prompt:

```python
user_prompt = "9.11 and 9.9 -- which is bigger?"
```

We can use `outlines.Generator` with the Pydantic class we previously defined, and call the generator with the Hermes prompt:

```python
generator = outlines.Generator(model, regex_str)
prompt = generate_hermes_prompt(json_schema=json_schema, user_prompt=user_prompt)
response = generator(prompt, max_tokens=1024, temperature=0, seed=42)
```

We obtain a series of intermediate reasoning steps as well as the conclusion:

```python
import json

json_response = json.loads(response)

print(json_response["reasoning"])
print(json_response["conclusion"])
# [{'reasoning_step': 'Both 9.11 and 9.9 are decimal numbers.'},
#  {'reasoning_step': 'When comparing decimal numbers, we look at the numbers after the decimal point.'},
#  {'reasoning_step': 'In this case, 9.11 has the number 1 after the decimal point, while 9.9 has the number 9.'},
#  {'reasoning_step': 'Since 1 is greater than 9, 9.11 is greater than 9.9.'}]
# '9.11 is bigger.'
```

We notice that the 4th reasoning step is wrong ``Since 1 is greater than 9, 9.11 is greater than 9.9.'', so we should probably give the model some examples for this particular task.

This example was originally contributed by [Alonso Silva](https://github.com/alonsosilvaallende).
