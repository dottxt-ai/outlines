# OpenAI and compatible APIs

!!! Installation

    You need to install the `openai` library to be able to use the OpenAI API in Outlines. Or alternatively:

    ```bash
    pip install "outlines[openai]"
    ```

## OpenAI models

Outlines supports models available via the OpenAI Chat API, e.g. GPT-4o, ChatGPT and GPT-4. You can initialize the model by passing the model name to `outlines.models.openai`:

```python
from outlines import models


model = models.openai("gpt-4o-mini")
model = models.openai("gpt-4o")
```

Check the [OpenAI documentation](https://platform.openai.com/docs/models/gpt-4o) for an up-to-date list of available models. You can pass any parameter you would pass to `openai.AsyncOpenAI` as keyword arguments:

```python
import os
from outlines import models


model = models.openai(
    "gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"]
)
```

The following table enumerates the possible parameters. Refer to the [OpenAI SDK's code](https://github.com/openai/openai-python/blob/54a5911f5215148a0bdeb10e2bcfb84f635a75b9/src/openai/_client.py) for an up-to-date list.

**Parameters:**

| **Parameters** | **Type** | **Description** | **Default** |
|----------------|:---------|:----------------|:------------|
| `api_key` | `str` | OpenAI API key. Infered from `OPENAI_API_KEY` if not specified | `None` |
| `organization` | `str` | OpenAI organization id. Infered from `OPENAI_ORG_ID` if not specified | `None` |
| `project` | `str` | OpenAI project id. Infered from `OPENAI_PROJECT_ID` if not specified.| `None` |
| `base_url` | `str | https.URL` | Base URL for the endpoint. Infered from `OPENAI_BASE_URL` if no specified. | `None` |
| `timeout` | `float` | Request timeout.| `NOT_GIVEN` |
| `max_retries` | `int` | Maximum number of retries for failing requests | `2` |
| `default_headers` | `Mapping[str, str]` | Default HTTP headers | `None` |
| `default_query` | `Mapping[str, str]` | Custom parameters added to the HTTP queries | `None` |
| `http_client` | `https.AsyncClient` | User-specified `httpx` client | `None` |

## Azure OpenAI models

Outlines also supports Azure OpenAI models:

```python
from outlines import models


model = models.azure_openai(
    "azure-deployment-name",
    "gpt-4o-mini",
    api_version="2024-07-18",
    azure_endpoint="https://example-endpoint.openai.azure.com",
)
```

!!! Question "Why do I need to specify model and deployment name?"

    The model name is needed to load the correct tokenizer for the model. The tokenizer is necessary for structured generation.


You can pass any parameter you would pass to `openai.AsyncAzureOpenAI`. You can consult the [OpenAI SDK's code](https://github.com/openai/openai-python/blob/54a5911f5215148a0bdeb10e2bcfb84f635a75b9/src/openai/lib/azure.py) for an up-to-date list.

**Parameters:**


| **Parameters** | **Type** | **Description** | **Default** |
|----------------|:---------|:----------------|:------------|
| `azure_endpoint` | `str` | Azure endpoint, including the resource. Infered from `AZURE_OPENAI_ENDPOINT` if not specified | `None` |
| `api_version` | `str` | API version. Infered from `AZURE_OPENAI_API_KEY` if not specified | `None` |
| `api_key` | `str` | OpenAI API key. Infered from `OPENAI_API_KEY` if not specified | `None` |
| `azure_ad_token` | `str` | Azure active directory token. Inference from `AZURE_OPENAI_AD_TOKEN` if not specified | `None` |
| `azure_ad_token_provider` | `AzureADTokenProvider` | A function that returns an Azure Active Directory token | `None` |
| `organization` | `str` | OpenAI organization id. Infered from `OPENAI_ORG_ID` if not specified | `None` |
| `project` | `str` | OpenAI project id. Infered from `OPENAI_PROJECT_ID` if not specified.| `None` |
| `base_url` | `str | https.URL` | Base URL for the endpoint. Infered from `OPENAI_BASE_URL` if not specified. | `None` |
| `timeout` | `float` | Request timeout.| `NOT_GIVEN` |
| `max_retries` | `int` | Maximum number of retries for failing requests | `2` |
| `default_headers` | `Mapping[str, str]` | Default HTTP headers | `None` |
| `default_query` | `Mapping[str, str]` | Custom parameters added to the HTTP queries | `None` |
| `http_client` | `https.AsyncClient` | User-specified `httpx` client | `None` |

## Models that follow the OpenAI standard

Outlines supports models that follow the OpenAI standard. You will need to initialize the OpenAI client properly configured and pass it to `outlines.models.openai`

```python
import os
from openai import AsyncOpenAI
from outlines import models
from outlines.models.openai import OpenAIConfig


client = AsyncOpenAI(
    api_key=os.environ.get("PROVIDER_KEY"),
    base_url="http://other.provider.server.com"
)
config = OpenAIConfig("model_name")
model = models.openai(client, config)
```

!!! Warning

    You need to pass the async client to be able to do batch inference.

## Structured Generation Support

Outlines provides support for [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs/json-mode) via `outlines.generate.json`, `outlines.generate.choice`

```python
from pydantic import BaseModel, ConfigDict
import outlines.models as models
from outlines import generate

model = models.openai("gpt-4o-mini")

class Person(BaseModel):
    model_config = ConfigDict(extra='forbid')  # required for openai
    first_name: str
    last_name: str
    age: int

generate.json(model, Person)
generator("current indian prime minister on january 1st 2023")
# Person(first_name='Narendra', last_name='Modi', age=72)

generator = generate.choice(model, ["Chicken", "Egg"])
print(generator("Which came first?"))
# Chicken
```

!!! Warning

    Structured generation support only provided to OpenAI-compatible endpoints which conform to OpenAI's standard. Additionally, `generate.regex` and `generate.cfg` are not supported.


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
config = OpenAIConfig("model_name")
model = models.openai(client, config)
```

It is possible to specify the values for `seed`, `presence_penalty`, `frequence_penalty`, `top_p` by passing an instance of `OpenAIConfig` when initializing the model:

```python
from outlines.models.openai import OpenAIConfig
from outlines import models


config = OpenAIConfig(
    presence_penalty=1.,
    frequency_penalty=1.,
    top_p=.95,
    seed=0,
)
model = models.openai("gpt-4o-mini", config)
```

## Monitoring API use

It is important to be able to track your API usage when working with OpenAI's API. The number of prompt tokens and completion tokens is directly accessible via the model instance:

```python
from openai import AsyncOpenAI
import outlines.models


model = models.openai("gpt-4o")

print(model.prompt_tokens)
# 0

print(model.completion_tokens)
# 0
```

These numbers are updated every time you call the model.
