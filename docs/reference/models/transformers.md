# Transformers

!!! Note "Documentation"

    To be able to use Transformers models in Outlines, you must install the `transformers` library, `pip install transformers`

    Consult the [`transformers` documentation](https://huggingface.co/docs/transformers/en/index) for detailed informations about how to initialize models and the available options.

## Create a `Transformers` model

You can use `outlines.from_transformers` to load a `transformers` model and tokenizer:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import models

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = models.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)
```

The model name must be a valid `transformers` model name. You can find a list of all in the HuggingFace library [here](https://huggingface.co/models). We currently support `CausalLM`, `Seq2Seq`, `Mamba` and vision models.

Be cautious with model selection though, some models such as `t5-base` don't include certain characters (`{`) and you may get an error when trying to perform structured generation.

## Use the model to generate text

Once you have created a `Transformers` model, you can use it to generate text by calling the instance of the model.

```python
model("Hello, how are you?")
```

You can also first create a `Generator` and then call it.
```python
from outlines import Generator

generator = Generator(model)
generator("Hello, how are you?")
```

`Transformers` models typically support batching and the generation of several samples at once.

For instance:
```python
model(["Hello, how are you?", "Respond with one word. Not more."], num_return_sequences=2, num_beams=2)
```

This would generate two sequences for each prompt, for a total of four sequences (two lists of 2 elements each in a list).

## Use the model to generate structured data

`Transformers` models can generate structured data by providing a value for the parameter `output_type` (the second positional argument of the `generate` method, right after the prompt).

Supported types include `Json`, `Choice`, `Regex` and `CFG`.

For instance:
```python
from outlines.types import Json
from pydantic import BaseModel

class Character(BaseModel):
    name: str

model("Create a character with a name.", Json(Character))
```
