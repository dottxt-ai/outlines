# OpenAI and compatible APIs

!!! Installation

    You need to install the `openai` library to be able to use the OpenAI API in Outlines. Or alternatively you can run:

    ```bash
    pip install "outlines[openai]"
    ```

## OpenAI models

Outlines supports models available via the OpenAI Chat API, e.g. GPT-4o, ChatGPT and GPT-4. You can initialize the model by passing the model name to `outlines.models.OpenAI`:

```python
from outlines import models


model = models.OpenAI("gpt-4o-mini")
model = models.OpenAI("gpt-4o")
```

Check the [OpenAI documentation](https://platform.openai.com/docs/models/gpt-4o) for an up-to-date list of available models. You can pass any parameter you would pass to `openai.OpenAI` as keyword arguments:

```python
import os
from outlines import models


model = models.OpenAI(
    "gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"]
)
```

Refer to the [OpenAI SDK's code](https://github.com/openai/openai-python/blob/54a5911f5215148a0bdeb10e2bcfb84f635a75b9/src/openai/_client.py) for an up-to-date list of the initialization parameters.

## Azure OpenAI models

Outlines also supports Azure OpenAI models:

```python
from outlines import models


model = models.AzureOpenAI(
    "azure-deployment-name",
    api_version="2024-07-18",
    azure_endpoint="https://example-endpoint.openai.azure.com",
)
```

You can pass any parameter you would pass to `openai.AzureOpenAI`. You can consult the [OpenAI SDK's code](https://github.com/openai/openai-python/blob/54a5911f5215148a0bdeb10e2bcfb84f635a75b9/src/openai/lib/azure.py) for an up-to-date list.

## Advanced configuration

For more advanced configuration option, such as support proxy, please consult the [OpenAI SDK's documentation](https://github.com/openai/openai-python):


```python
from openai import AsyncOpenAI, DefaultHttpxClient
from outlines import models
from outlines.models.openai import OpenAIConfig


client = AsyncOpenAI(
    base_url="http://my.test.server.example.com:8083",
    http_client=DefaultHttpxClient(
        proxies="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

## Models that follow the OpenAI standard

Outlines supports models that follow the OpenAI standard. You will need to initialize the OpenAI client properly configured and pass it to `outlines.models.OpenAI`

```python
import os
from openai import AsyncOpenAI
from outlines import models
from outlines.models.openai import OpenAIConfig


model = models.OpenAI(
    "model_name",
    api_key=os.environ.get("PROVIDER_KEY"),
    base_url="http://other.provider.server.com"
)
```

## Text generation

To generate text using an OpenAI model you need to build a `Generator` object, possibly with the desired output type. You can then call the model by calling the `Generator`. The method accepts every argument that you could pass to the `client.completions.create` function, as keyword arguments:

```python
from outlines import models, Generator

model = models.OpenAI("gpt-4o-mini")
generator = Generator(model)
result = generator("Prompt", seed=10)
```

See the [OpenAI SDK documentation](https://github.com/openai/openai-python/blob/6974a981aec1814b5abba429a8ea21be9ac58538/src/openai/types/completion_create_params.py#L13) for the list of available arguments.

### Structured Generation Support

Outlines provides support for [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs/json-mode). Currently only JSON-Schema is supported:

```python
from pydantic import BaseModel
from outlines import models, Generator
from outlines.types import Json

model = models.OpenAI("gpt-4o-mini")

class Person(BaseModel):
    first_name: str
    last_name: str
    age: int

generator = Generator(model, Json(Person))
generator("current indian prime minister on january 1st 2023")
# Person(first_name='Narendra', last_name='Modi', age=72)
```

The following objects can be used to define the structure of the Json object:
- A string that represents a Json Schema
- A dictionary that represents a Json Schema
- A Pydantic model
- A TypedDict
- A dataclass
