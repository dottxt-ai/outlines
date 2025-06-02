# OpenAI and compatible APIs

!!! Note "Documentation"

    To be able to use OpenAI in Outlines, you must install the OpenAI Python SDK with `pip install openai`

    Consult the [OpenAI SDK documentation](https://github.com/openai/openai-python) for detailed informations about how to initialize OpenAI clients and the available options.

## OpenAI models

Outlines supports models available via the OpenAI Chat API, e.g. GPT-4o, ChatGPT and GPT-4. You can initialize the model by passing the model name to `outlines.models.OpenAI`:

```python
from openai import OpenAI
import outlines

# OpenAI models
client = OpenAI()
model = outlines.from_openai(client, "gpt-4o-mini")
model = outlines.from_openai(client, "gpt-4o")

# OpenAI models deployed on Azure
client = AzureOpenAI(
    api_version="2024-07-18",
    azure_endpoint="https://example-endpoint.openai.azure.com",
)
model = outlines.from_openai(client, "azure-deployment-name")
```

Check the [OpenAI documentation](https://platform.openai.com/docs/models/gpt-4o) for an up-to-date list of available models, and the [OpenAI SDK's code](https://github.com/openai/openai-python/blob/54a5911f5215148a0bdeb10e2bcfb84f635a75b9/src/openai/_client.py) for an up-to-date list of the initialization parameters.


## Advanced configuration

For more advanced configuration option, such as support proxy, please consult the [OpenAI SDK's documentation](https://github.com/openai/openai-python):


```python
from openai import AsyncOpenAI, DefaultHttpxClient
import outlines

client = AsyncOpenAI(
    api_key="my key",
    base_url="http://my.test.server.example.com:8083",
    http_client=DefaultHttpxClient(
        proxies="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
model = outlines.from_openai(client, "model_name")
```

## Models that follow the OpenAI standard

Outlines supports models that follow the OpenAI standard. You will need to initialize the OpenAI client properly configured and pass it to `outlines.models.OpenAI`

```python
import os
from openai import AsyncOpenAI
import outlines

client = AsyncOpenAI(
    api_key=os.environ.get("PROVIDER_KEY"),
    base_url="http://other.provider.server.com",
)
model = outlines.from_openai(client, "model_name")
```

## Calling the model

You can call the model directly. The method accepts every argument that you could pass to the `client.completions.create` function, as keyword arguments:

```python
from openai import OpenAI
import outlines

model = outlines.from_openai(OpenAI(), "gpt-4o-mini")
result = model("Prompt", seed=10)
```

See the [OpenAI SDK documentation](https://github.com/openai/openai-python/blob/6974a981aec1814b5abba429a8ea21be9ac58538/src/openai/types/completion_create_params.py#L13) for the list of available arguments.

### Structured Generation Support

Outlines provides support for [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs/json-mode). Currently only JSON-Schema is supported:

```python
from openai import OpenAI
from pydantic import BaseModel

import outlines

model = outlines.from_openai(OpenAI(), "gpt-4o-mini")

class Person(BaseModel):
    first_name: str
    last_name: str
    age: int

generator = Generator(model, Person)
generator("current indian prime minister on january 1st 2023")
# Person(first_name='Narendra', last_name='Modi', age=72)
```

The following objects can be used to define the structure of the Json object:
- A string that represents a Json Schema
- A dictionary that represents a Json Schema
- A Pydantic model
- A TypedDict
- A dataclass
