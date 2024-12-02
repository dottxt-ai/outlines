# Gemini

!!! Installation

    You need to install the `google-generativeai` library to be able to use the Gemini API in Outlines. Or alternatively you can run:

    ```bash
    pip install "outlines[gemini]"
    ```

## Gemini models

Outlines supports models available via the Gemini API, e.g. Gemini 1.5. You can initialize the model by passing the model name to `outlines.models.Gemini`:

```python
from outlines import models

model = models.Gemini("gemini-1-5-flash")
model = models.Gemini("gemini-1-5-pro")
```

Check the [Gemini documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for an up-to-date list of available models.

## Text generation

To generate text using a Gemini model you need to build a `Generator` object, possibly with the desired output type. You can then call the model by calling the `Generator`. The method accepts every argument that you could pass to the `client.completions.create` function, as keyword arguments:

```python
from outlines import models, Generator

model = models.Gemini("gemini-1-5-flash")
generator = Generator(model)
result = generator("Prompt", max_tokens=1024)
```

### Structured generation

Gemini provides support for structured outputs.

#### Json Schema

Outlines provides support for JSON Schema-based structured generation with the Gemini models:

```python
from collections import TypedDict
from outlines import Generator, models
from outlines.types import Json

model = models.Gemini("gemini-1-5-flash")

class Person(TypedDict):
    first_name: str
    last_name: str
    age: int

generator = Generator(model, Json(Person))
generator("current indian prime minister on january 1st 2023")
# Person(first_name='Narendra', last_name='Modi', age=72)
```

Because of the current limitations of the Gemini SDK only The following objects can be used to define the structure of the Json object:
- A Pydantic model
- A TypedDict

#### Multiple choices

Outlines provides support for multiple-choices structured generation. Enums and lists of choices are supported:

```python
from enum import Enum
from outlines import Generator, models
from outlines.types import Choice

model = models.Gemini("gemini-1-5-flash")

class Foo(Enum):
    foo = "Foo"
    fizz = "Fizz"
    fuzz = "Fuzz"

generator = Generator(model, Choice(Foo))
generator("current indian prime minister on january 1st 2023")
# Person(first_name='Narendra', last_name='Modi', age=72)
```

The following objects can be used to define the choices:
- An Enum object
- A Python list
