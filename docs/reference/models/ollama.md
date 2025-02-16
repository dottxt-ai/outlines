# Ollama

!!! Installation

    To be able to use Ollama in Outlines, you must install both Ollama and the `ollama` python sdk.

    - To download Ollama: https://ollama.com/download
    - To install the ollama python sdk: `pip install ollama`

## Ollama models

You must provide a model name when instantiating the `outlines.models.Ollama` class. This model must be available on your system.
```python
from outlines.models import Ollama

model = Ollama("tinyllama")
```

To download a new model from the Ollama model hub, you can use the following command (it will return an `Ollama` instance):
```python
from outlines.models import Ollama

Ollama.from_pretrained("llama3.1:8b")
```

You can find the list of available models on the [Ollama library](https://ollama.com/library).

## Generate text

As with other models, you can either first create a `Generator` object and then call it
```python
from outlines.models import Ollama
from outlines.generate import Generator

model = Ollama("tinyllama")
generator = Generator(model)
answer = generator("Write a sentence about a cat.")
```
or directly call the model
```python
from outlines.models import Ollama

model = Ollama("tinyllama")
answer = model("Write a sentence about a cat.")
```

The input of the generation must be a string. Batch generation is not supported.
The only output type supported is `Json`.
```python
from outlines.models import Ollama
from outlines.types import Json
from pydantic import BaseModel

class Character(BaseModel):
    name: str

model = Ollama("tinyllama")
answer = model("Create a character.", output_type=Json(Character))
```

You can also stream the tokens:
```python
from outlines.models import Ollama

model = Ollama("tinyllama")
tokens = model.stream("Write a sentence about a cat.")
```

## Optional parameters

You can provide the same optional parameters you would pass to the `ollama` sdk's client both during the initialization of the `Ollama` class and when generating text.
Consult the [ollama python sdk Github repository](https://github.com/ollama/ollama-python) for the full list of parameters.
