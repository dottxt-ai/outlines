# Transformers


!!! Installation

    You need to install the `transformer` library to be able to use these models in Outlines, or alternatively:

    ```bash
    pip install "outlines[transformers]"
    ```

## Create a `Transformers` model

The only mandatory argument to instantiate a `Transformers` model is the name of the model to use.
```python
from outlines import models

model = models.Transformers("microsoft/Phi-3-mini-4k-instruct")
```

The model name must be a valid `transformers` model name. You can find a list of all in the HuggingFace library [here](https://huggingface.co/models).

When instantiating a `Transformers` model as such, the class creates a model from the transformers libray using the class `AutoModelForCausalLM` by default (`transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)`).

You can also provide keyword arguments in an optional `model_kwargs` parameter. Those will be passed to the `from_pretrained` method of the model class. One such argument is `device_map`, which allows you to specify the device on which the model will be loaded.

For instance:
```python
from outlines import models

model = models.Transformers("microsoft/Phi-3-mini-4k-instruct", model_kwargs={"device_map": "cuda"})
```

## Alternative model classes

If the model you want to use is not compatible with `AutoModelForCausalLM`, you must provide a value for the `model_class` parameter. This value must be a valid `transformers` model class.

For instance:
```python
from outlines import models
from transformers import AutoModelForSeq2SeqLM

model = models.Transformers("facebook/bart-large", model_class=AutoModelForSeq2SeqLM)
```

When you instantiate a `Transformers` model, the class also creates a `Tokenizer` instance from the `AutoTokenizer` class. You can provide keyword arguments in an optional `tokenizer_kwargs` parameter. Those will be passed to the `from_pretrained` method of the tokenizer class as such: `tokenizer_class.from_pretrained(model_name, **tokenizer_kwargs)`.

Similarly, if your model is not compatible with `AutoTokenizer`, you must provide a value for the `tokenizer_class` parameter.

```python
from outlines import models
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_pile_t5 = models.Transformers(
    model_name="EleutherAI/pile-t5-large",
    model_class=T5ForConditionalGeneration,
    tokenizer_class=T5Tokenizer
)
```

### Mamba

[Mamba](https://github.com/state-spaces/mamba) is a transformers alternative which employs memory efficient, linear-time decoding.

To use Mamba with outlines you must first install the necessary requirements:
```
pip install causal-conv1d>=1.2.0 mamba-ssm torch transformers
```

Then you can create an `Mamba` Outlines model via:
```python
from outlines import models

model = models.Mamba("state-spaces/mamba-2.8b-hf", model_kwargs={"device_map": "cuda"}, tokenizer_kwargs={"padding_side": "left"})
```

Alternatively, you can use the `Transformers` class to create an `Mamba` model by providing the appropriate `model_class` and `tokenizer_class` arguments.

Read [`transformers`'s documentation](https://huggingface.co/docs/transformers/en/model_doc/mamba) for more information.

### Encoder-Decoder Models

You can use encoder-decoder (seq2seq) models like T5 and BART with Outlines.

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
