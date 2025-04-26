# vLLM

## Prerequisites

The Outlines `VLLM` model is inteded to be used along with a vllm instance running on a separate server (can be local or remote). Make sure you have a vllm server running before using the `VLLM` model. As the vllm client relies on the `openai` python sdk, you need to have an `openai` package installed. If you instead want to use the vllm offline inference mode, please refer to the [VLLMOffline model documentation](./vllm_offline.md).

## Initialize the model

To load the model, you can use the `from_vllm` function. The argument of the function is either an `OpenAI` or `AsyncOpenAI` instance from the `openai` library. Based on whether the `openai` instance is synchronous or asynchronous, you will receive a `VLLM` or `AsyncVLLM` model instance.

```python
import openai
import outlines

sync_openai_client = openai.OpenAI(base_url="...")
async_openai_client = openai.AsyncOpenAI(base_url="...")

sync_model = outlines.from_vllm(sync_openai_client)
print(type(sync_model)) # <class 'outlines.models.vllm.VLLM'>

async_model = outlines.from_vllm(async_openai_client)
print(type(async_model)) # <class 'outlines.models.vllm.AsyncVLLM'>
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

The `VLLM` model supports also supports streaming.

```python
for chunk in sync_model.stream("Write a short story about a cat.", max_tokens=100):
    print(chunk)
```

## Optional parameters

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `openai` client. An optional parameter of particular interest is `extra_body`, which is a dictionnary containing arguments that are specific to vLLM and are not part of the standard `openai` interface (see the [vLLM documentation][https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html] on the OpenAI-compatible server for more information).

[vllm-docs]: https://docs.vllm.ai/en/latest/
[vllm-online-quickstart]: https://docs.vllm.ai/en/latest/getting_started/quickstart.html#quickstart-online
