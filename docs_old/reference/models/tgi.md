# TGI

## Prerequisites

The Outlines `TGI` model is intended to be used along with a HuggingFace `Text Generation Inference` server (running locally or remotely). Make sure you have a TGI server running before using the `TGI` model. As the TGI client relies on the `huggingface_hub` python package, you need to have it installed.

## Initialize the model

    To load the model, you can use the `from_tgi` function. The argument of the function is an instance of `InferenceClient` or `AsyncInferenceClient` from the `huggingface_hub` package. You will respectively receive a `TGI` or `AsyncTGI` model instance.

```python
import outlines
from huggingface_hub import InferenceClient, AsyncInferenceClient

sync_model = outlines.from_tgi(InferenceClient("http://localhost:8080"))
print(type(sync_model))  # <class 'outlines.models.tgi.TGI'>

async_model = outlines.from_tgi(AsyncInferenceClient("http://localhost:8080"))
print(type(async_model))  # <class 'outlines.models.tgi.AsyncTGI'>
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

The `TGI` model also supports streaming:

```python
for chunk in sync_model.stream("Write a short story about a cat.", max_new_tokens=100):
    print(chunk)
```

## Optional parameters

When calling the model, you can provide optional parameters on top of the prompt and the output type. These parameters will be passed directly to the TGI client. Common parameters include `max_new_tokens`, `stream`, `temperature`, `top_k`, `top_p`, and others as specified in the [TGI inference client documentation][tgi-inference-client-docs].


[tgi-docs]: https://huggingface.co/docs/text-generation-inference/en/index
[tgi-inference-client-docs]: https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
[tgi-structured-output-docs]: https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/using_guidance
