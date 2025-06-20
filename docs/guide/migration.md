# Outlines 1.0 migration guide

Outlines 1.0 introduces some breaking changes that affect the way you use the library. You are likely concerned by all of the following sections, so please read this document carefully until the end.

This guide will help you migrate your code to the new version.

All previous functionalities will be supported until Outlines version 1.1.0, but a warning message will be displayed to remind you to migrate your code and provide instructions to help you do so. Please migrate your code to the v1 as soon as possible.

## Removed or modified features
- [Generate functions](#generate-functions)
- [Models](#models)
- [Samplers](#samplers)
- [Functions](#functions)
- [Text generation return types](#text-generation-return-types)
- [Inference arguments](#inference-arguments)

### Generate functions

The whole `generate` module has been removed. That includes the functions `generate.cfg`, `generate.choice`, `generate.format`,`generate.fsm`, `generate.json`, `generate.regex` and `generate.text`.

You should replace these functions by the [`Generator`](../features/core/generator.md) object along with the right output type as an argument (on top of the model). The output type can either be a python type or be an object from the `outlines.types` module. You can find more information about the output types in the [Output Types](../features/core/output_types.md) section of the features documentation.

Associated v1 output types for each deprecated function:
- `generate.cfg` -> `outlines.types.CFG`
- `generate.choice` -> `typing.Literal` or `typing.Union`
- `generate.format` -> native python types (`str`, `int` etc.)
- `generate.fsm` -> `outlines.types.FSM`
- `generate.json` -> `pydantic.BaseModel`, `typing.TypedDict`, `dataclasses.dataclass`, `genson.schema.SchemaBuilder` or `outlines.types.JsonSchema`
- `generate.regex` -> `outlines.types.Regex`
- `generate.text` -> no output type (`None`)

For instance, instead of:

```python
from outlines import generate

model = ...
generator = generate.choice(model, ["foo", "bar"])
```

You should now use:

```python
from typing import Literal
from outlines import Generator

model = ...
generator = Generator(model, Literal["foo", "bar"])
```

### Models

The model classes found in the `outlines.models` module are maintained but there are a few important changes to be aware of.

The functions used to created a model have been replaced by equivalent functions named with a `from_` prefix. The function `outlines.models.transformers` has been replaced by `outlines.from_transformers` for instance. On top of this change of name, the arguments have been modified. You should refer to the [models documentation](../features/models/index.md) for more details, but the overall idea is that you now need to provide a model/client instance from the inference library the Outlines model is wrapping.

For instance, instead of:

```python
from outlines import models

model = models.llamacpp(
    repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
    filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
)
```

You should now do:

```python
from llama_cpp import Llama
from outlines import from_llamacpp

llamacpp_model = Llama.from_pretrained(
    repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
    filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
)
model = from_llamacpp(llamacpp_model)
```

The `load_lora` methods that are present on the `VLLM` and `LlamaCpp` models have been removed. You should now handle lora loading through the `Llama` instance in the case of the `LlamaCpp` model or provide it as a keyword argument when calling the model in the case of the `VLLM` model.

For instance, instead of:

```python
from outlines import from_vllm
from vllm import LLM

model = from_vllm(
    LLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
)
model.load_lora("path/to/lora/file")

response = model("foo")
```

You should now do:

```python
from outlines import from_vllm
from vllm import LLM
from vllm.lora.request import LoRARequest

model = from_vllm(
    LLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
)
lora_request = LoRARequest("path/to/lora/file", 1, "path/to/lora/file")

response = model("foo", lora_request=lora_request)
```

The `ExLlamav2` model has been removed without replacement. This inference library is not fully compatible with Outlines, so we decided to remove it. You can still use it until final deprecation, but we recommend you to migrate to a different inference library right now.

### Samplers

The `outlines.samplers` module has been removed without replacement. You should now use the arguments of the inference library model to control the sampling. Depending on the model you use, this could be done at initialization or when calling the model to generate text (so when calling the outlines model or a generator).

For instance, instead of:

```python
from outlines import generate

model = <transformers_model>

generator = generate.text(model, samplers.beam_search(2))
response = generator("foo")
```

You should now do:

```python
from outlines import Generator

model = <transformers_model>

generator = Generator(model)
response = generator("foo", num_beams=2)
```

### Functions

The `outlines.function` module has been removed. It is replaced by the `outlines.applications` module. An [`Application`](../features/utility/application.md) serves a similar purpose as a `Function`: it encapsulates a prompt template and an output type. A difference is that can `Application` is not instantiated with a model name. Instead, you should provide a model instance along with the prompt when calling it.

For instance, instead of:

```python
from outlines import Function

prompt_template = ...
output_type = ...

fn = Function(
    prompt_template,
    output_type,
    "hf-internal-testing/tiny-random-GPTJForCausalLM",
)

result = fn("foo")
```

You should now do:

```python
from outlines import Application

prompt_template = ...
output_type = ...

application = Application(
    prompt_template,
    output_type,
)

model = ...

result = application(model, "foo")
```

### Text generation return types

In the previous version of Outlines, the return type of the generators depended on the output type provided. For instance, if you passed a Pydantic model to the `generate.json` function, the return type was a Pydantic model instance. In the v1, the return type of a generator is always a `str`, the raw text generated by the model. You are responsible for parsing the text into the desired format.

For instance, instead of:

```python
from pydantic import BaseModel
from outlines import generate

class Foo(BaseModel):
    bar: str

model = ...

generator = generate.json(model, Foo)

result = generator("foo")
print(result.bar)
```

You should now do:

```python
from pydantic import BaseModel
from outlines import Generator

class Foo(BaseModel):
    bar: str

model = ...

generator = Generator(model, Foo)

result = generator("foo")
result = Foo.model_validate_json(result) # parse the text into the Pydantic model instance
print(result.bar)
```

The [Output Types](../features/core/output_types.md) section of the features documentation includes extensive details on available output types.

### Inference arguments

In the previous version of Outlines, some of the inference arguments were standardized across the models and were provided as positional arguments to the generator or through the sampling params dictionary. Additionally, various default values were added by outlines to the inference library models. This is no longer the case. You should refer to the documentation of the inference library you use to find the right arguments for your use case and pass them as keyword arguments to the outlines generator when calling it.

For instance, instead of:

```python
from outlines import generate

model = <transformers_model>

generator = generate.text(model)

result = generator("foo", 256, ".", 10) # 256 tokens, stop at "." and seed 10
```

You should now do:

```python
from outlines import Generator

model = <transformers_model>

generator = Generator(model)

result = generator("foo", max_new_tokens=256, stop_strings=".", seed=10)
```
