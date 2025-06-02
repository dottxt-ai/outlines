# Dottxt

!!! Installation

    To be able to use Dottxt in Outlines, you must install the `dottxt` python sdk.

    ```bash
    pip install dottxt
    ```

    You also need to have a Dottxt API key. This API key must either be set as an environment variable called `DOTTXT_API_KEY` or be provided to the `outlines.models.Dottxt` class when instantiating it.

## Generate text

Dottxt only supports constrained generation with the `Json` output type. The input of the generation must be a string. Batch generation is not supported.
Thus, you must always provide an output type.

You can either create a `Generator` object and call it afterward:

```python
from outlines.models import Dottxt
from outlines import Generator
from pydantic import BaseModel

class Character(BaseModel):
    name: str

model = Dottxt()
generator = Generator(model, Character)
result = generator("Create a character")
```

or call the model directly with the output type:

```python
from outlines.models import Dottxt
from pydantic import BaseModel

class Character(BaseModel):
    name: str

model = Dottxt()
result = model("Create a character", Character)
```

In any case, compilation for a given output type happens only once (the first time it is used to generate text).

## Optional parameters

You can provide the same optional parameters you would pass to the `dottxt` sdk's client both during the initialization of the `Dottxt` class and when generating text.
Consult the [dottxt python sdk Github repository](https://github.com/dottxt-ai/dottxt-python) for the full list of parameters.
