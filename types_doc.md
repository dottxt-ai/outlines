---
title: Output Types
---

# Output Types

Outlines provides a simple and intutive way of defining the output format of text generation.

## General approach

The general idea is that you should provide as an output type what you would give as the type hint of the return type of a function.
Thus, you can use most Python native types as well as the additional types from the `typing` module and some widespread Python packages such as `pydantic`.

Consider the following functions for instance:

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel

class Baz(BaseModel):
    foo: datetime
    bar: Union[Dict, List[str]]

def foo() -> int:
    ...

def bar() -> Literal["a", "b"]:
    ...

def baz() -> Baz:
    ...
```

With an Outlines generator, you can generate text that would respect the type hints above by providing those as the output type:

```python
generator("my prompt", int) # "12"
generator("my prompt", iteral["a", "b"]) # "12"
generator("my prompt", Baz) # "{'foo': '', 'bar': ['qux', 'quux']}"
```

An important difference with function type hints though is that an Outlines generator always returns a string.
You have to cast the response into the type you want yourself.

## Outlines-specific types

While Outlines favors the widely adopted types described, there are a few specific cases in which you need to use a type define in the `outlines.types` module. Those types are necessary to avoid ambiguities and to respect the principle that out output types could be used as return type hints of functions.

### JsonSchema

There are multiple types you can use to defining a json schema-like structured output:
- A Pydantic class
- A dataclass
- A TypedDict
- A GenSon SchemaBuilder

However, 2 populars format are not supported directly: json schema strings and python dictionaries describing a json schema. To use those, you should wrap them in a `JsonSchema` object. For instance :

```python
from outlines.types import JsonSchema

schema_string = '{"type": "object", "properties": {"answer": {"type": "number"}}}'
output_type = JsonSchema(schema_string)

schema_dict = {
    "type": "object",
    "properties": {
        "answer": {"type": "number"}
    }
}
output_type = JsonSchema(schema_dict)
```

### Regex

As regex patterns are expressed as simple raw string literals, we also need to wrap them in an Outlines object: `outlines.types.Regex`. For instance:

```python
from outlines.types import Regex

regex = r"[0-9]{3}"
output_type = Regex(regex)
```

### Context-Free Grammar

Outlines allows you to generate text that respects the syntax of a context-free grammar. For that, you need to define a Lark grammar. As the latter is expressed as a string, we need once again to wrap in an Outlines object: `outlines.types.CFG`. For instance:

```python
from outlines.types import CFG

grammar_string = """
    start: expr
    expr: "{" expr "}" | "[" expr "]" |
"""
output_type = CFG(grammar_string)
```

## Output type availability

The output types presented above are not available for all models as some have only limited support for structured outputs. In general, we need to distinguish two different types of models:
- steerable models: they are fully run locally without a separate server such as that Outlines can control the structured generation process through the creation of a `LogitsProcessor`. For those, all output types available in Outlines are supported. Steerable models are `Transformers`, `VLLMOffline`, `LlamaCpp` and `MLXLM`.
- blackbox models: Outlines interact with a client but text generation happen in a separate server (either remote or local). In that case, Outlines can provide structured generations arguments as defined by the server's API, but it does not control the generation. Specific output type support depends on each model, please consult the model's documentation for more information. All models that are not listed in the steerable models fall in that category such as `OpenAI`, `VLLM`, `Ollama`...

