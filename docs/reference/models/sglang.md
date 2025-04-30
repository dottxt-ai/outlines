# SgLang

## Prerequisites

The Outlines `SgLang` model is inteded to be used along with a separate sglang server (running either locally or remotely). Make sure you have an sglang server running before using the `SgLang` model. As the sglang client relies on the `openai` python sdk, you need to have the `openai` package installed.

## Initialize the model

To load the model, you can use the `from_sglang` function. The argument of the function is either an `OpenAI` or `AsyncOpenAI` instance from the `openai` library. Based on whether the `openai` instance is synchronous or asynchronous, you will receive a `SgLang` or `AsyncSgLang` model instance.

```python
import openai
import outlines

sync_openai_client = openai.OpenAI(base_url="...")
async_openai_client = openai.AsyncOpenAI(base_url="...")

sync_model = outlines.from_sglang(sync_openai_client, "qwen/qwen2.5-0.5b-instruct")
print(type(sync_model)) # <class 'outlines.models.sglang.SgLang'>

async_model = outlines.from_sglang(async_openai_client, "qwen/qwen2.5-0.5b-instruct")
print(type(async_model)) # <class 'outlines.models.sglang.AsyncSgLang'>
```

## Generate text

To generate text, you can call the model with a prompt as argument and optionally an output type to rely on structured generation:

```python
from pydantic import BaseModel

class Character(BaseModel):
    name: str

answer = sync_model("Create a character.", output_type=Character)
answer = await async_model("Create a character.", output_type=Character)
```

The `SgLang` model supports also supports streaming.

```python
for chunk in sync_model.stream("Write a short story about a cat.", max_tokens=100):
    print(chunk)
```

## Optional parameters

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `openai` client. An optional parameter of particular interest is `extra_body`, which is a dictionnary containing arguments that are specific to sglang and are not part of the standard `openai` interface (see the [sglang documentation][sglang-docs] on the OpenAI-compatible server for more information).

[sglang-docs]: https://docs.sglang.ai/start/install.html
[sglang-structured-output-quickstart]: https://docs.sglang.ai/backend/structured_outputs.html#JSON
