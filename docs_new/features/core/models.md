---
title: Models
---

# Models

Outlines models are objects that wrap an inference client or engine. Models provide a standardized interface to generate structured text.

Models available in Outlines:
- [Anthropic](../../models/anthropic)
- [Dottxt](../../models/dottxt)
- [Gemini](../../models/gemini)
- [LlamaCpp](../../models/llamacpp)
- [MLXLM](../../models/mlxlm)
- [Ollama](../../models/ollama)
- [OpenAI](../../models/openai)
- [SGLang](../../models/sglang)
- [TGI](../../models/tgi)
- [Transformers](../../models/transformers)
- [VLLM](../../models/vllm) (server-based inference)
- [VLLMOffline](../../models/vllm_offline) (offline inference)

!!! note

    The model loading functions have been modified in v1. While they used to be called `<name_inference_library>`, they are now called `from_<name_inference_library>`. The model classes' names and `__init__` methods are left unchanged.

## Overview

All Outlines model classes have an associated loader function to facilitate instantiating a model instance. The name of this function is `from_` plus the name of the model in lower-case letters. For instance, Outlines has a `Transformers` model and an associated `from_transformers` loader function. The parameters to load a model are specific to each provider, please consult the documentation of the model you want to use for more information.

Outlines models implement the following methods you can use to generate text:

### `__call__(model_input, output_type, **inference_kwargs)`

Generate text with the model for the given input and output type.

- **model_input**: The input to give to pass on to the provider. This is most often a prompt string.
- **output_type**: The output type to use to constrain the generation. Consult the documentation on output types for more information.
- **inference_kwargs**: Additional keyword arguments passed to the provider's text generation function.
- **Returns**: The generated text conforming to the specified output type.

### `stream(model_input, output_type, **inference_kwargs)`

Stream text generation with the model for the given input and output type.

- **model_input**: The input to give to pass on to the provider. This is most often a prompt string.
- **output_type**: The output type to use to constrain the generation. Consult the documentation on output types for more information.
- **inference_kwargs**: Additional keyword arguments passed to the provider's text generation function.
- **Returns**: The generated text conforming to the specified output type.

For instance:

```python
from outlines import from_transformers
import transformers

# Create a model
model = from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    transformers.AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

# Regular text generation
response = model("How many countries are there in the world", int, max_new_tokens=20)
print(result) # "200"

# Streaming
for chunk in generator.stream("Tell me a story", max_new_tokens=20):
    print(chunk) # Once...
```

Models can also be used to construct a generator through which you can generate text. Please consult the [generator documentation](../generator) for more information.

## Types of models

Models can be divided into two categories: local models and server-based models (the server could be local or remote).

In the case of local models, the text generation happens within the provider object used to instantite the model. This gives Outlines direct access to the generation process (through a logits processor) and means all structured generation output types are available.

In the case of server-based models, the provider object used to instantiate the model is a client that sends a request to a server that is in charge of the actual text generation. As a result, we have limited control over text generation and some output types are not supported. For example, OpenAI does not currently support regular expression structured generation.

Local models:
- LlamaCpp
- MLXLM
- Transformers
- VLLMOffline

Server-based models:
- Anthropic
- Dottxt
- Gemini
- Ollama
- OpenAI
- SgLang
- TGI
- VLLM

Some models have an async version. To use them, just pass the async version of the provider object to their loading function. It will then return a `Async<ModelName>` instance with the same methods and features as the regular sync instance.

For instance:

```python
from outlines import from_tgi
from huggingface_hub import AsyncInferenceClient

model = from_tgi(
    AsyncInferenceClient("http://localhost:8000/v1")
)
print(type(model)) # outlines.models.tgi.AsyncTGI
```

The models that have an async version are the following:
- SgLang
- TGI
- VLLM
