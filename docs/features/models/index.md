---
title: Models
---

# Models

## Overview

Outlines models are objects that wrap an inference client or engine. Models provide a standardized interface to generate structured text.

!!! Warning

    The model loading functions have been modified in v1. While they used to be called `<name_inference_library>`, they are now called `from_<name_inference_library>`. The model classes' names and `__init__` methods are left unchanged.


All Outlines model classes have an associated loader function to facilitate initializing a model instance. The name of this function is `from_` plus the name of the model in lower-case letters. For instance, Outlines has a `Transformers` model and an associated `from_transformers` loader function. The parameters to load a model are specific to each provider, please consult the documentation of the model you want to use for more information.

After having created a model instance, you can either directly call it to generate text or first create a reusable generator that you would then call. In either case, you can provide an `output_type` to constrain the format of the generation output. See the [output types section](../core/output_types.md) for more information on constrained generation.

For instance:

```python
from outlines import from_transformers, Generator
import transformers

# Create a model
model = from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    transformers.AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

# Call it directly
response = model("How many countries are there in the world", max_new_tokens=20)
print(result) # 'There are 200 countries in the world.'

# Call it directly with an output_type
response = model("How many countries are there in the world", int, max_new_tokens=20)
print(result) # '200'

# Create a generator first and then call it
generator = Generator(model, int)
response = generator("How many countries are there in the world")
print(result) # '200'
```

## Features Matrix

In alphabetical order:

| | [Anthropic](../../models/anthropic) | [Dottxt](../../models/dottxt) | [Gemini](../../models/gemini) | [LlamaCpp](../../models/llamacpp) | [MLXLM](../../models/mlxlm) | [Ollama](../../models/ollama) | [OpenAI](../../models/openai) | [SGLang](../../models/sglang) | [TGI](../../models/tgi) | [Transformers](../../models/transformers) | [Transformers MultiModal](../../models/transformers_multimodal) | [VLLM](../../models/vllm) | [VLLMOffline](../../models/vllm_offline) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Output Types** | | | | | | | | | | | | | |
| Simple Types | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| JSON Schema | ❌ | ✅ | 🟠 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multiple Choice | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Regex | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Grammar | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | 🟠 | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Generation Features** | | | | | | | | | | | | | |
| Async | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| Streaming | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| Vision | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Batching | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |

## Model Types

Models can be divided into two categories: local models and server-based models.

In the case of local models, the text generation happens within the inference library object used to instantiate the model. This gives Outlines direct access to the generation process (through a logits processor) and means all structured generation output types are available.

The local models available are the following:

- LlamaCpp
- MLXLM
- Transformers
- TransformersMultiModal
- VLLMOffline

In the case of server-based models, the model is initialized with a client that sends a request to a server that is in charge of the actual text generation. As a result, we have limited control over text generation and some output types are not supported. The server on which the text generation happens can either be remote (with OpenAI or Anthopic for instance) or local (with SGLang for instance).

The server-based models available are the following:

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

- Ollama
- SgLang
- TGI
- VLLM
