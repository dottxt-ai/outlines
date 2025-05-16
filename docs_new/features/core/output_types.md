---
title: Output Types
---

# Output Types

Outlines provides a simple and intutive way of defining the output format of text generation.

## Overview

The general idea is that you should provide as an output type what you would give as the type hint of the return type of a function.
Thus, you can use most Python native types as well as the additional types from the `typing` module and some widespread Python packages such as `pydantic`.

Consider the following functions for instance:

```python
from datetime import date
from typing import Dict, List, Literal, Union
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    birth_date: date
    skills: Union[Dict, List[str]]

def give_int() -> int:
    ...

def pizza_or_burger() -> Literal["pizza", "burger"]:
    ...

def create_character() -> Character:
    ...
```

With an Outlines model, you can generate text that would respect the type hints above by providing those as the output type:

```python
model("How many minutes are there in one hour", int) # "60"
model("Pizza or burger", Literal["pizza", "burger"]) # "pizza"
model("Create a character", Character, max_new_tokens=100)
```

An important difference with function type hints though is that an Outlines generator always returns a string.
You have to cast the response into the type you want yourself.

For instance:

```python
result = model("Create a character", Character, max_new_tokens=100)
casted_result = Character.model_validate_json(result)
print(result) # '{"name": "Aurora", "birth_date": "1990-06-15", "skills": ["Stealth", "Diplomacy"]}'
print(casted_result) # name=Aurora birth_date=datetime.date(1990, 6, 15) skills=['Stealth', 'Diplomacy']
```

## Output Type Categories

We can group possible output types in several categories based on the use case they correspond to. While most of those types are native python or types coming from well-known third-party libraries, there are three Outlines-specific types: `JsonSchema`, `Regex` and `CFG`. Their use is explained below.

### Basic Types

The most straightfoward form of structed generation is to return an answer that conforms to a given basic type such as an int or a python list. You can use the basic Python types, but also the types contained in the `typing` library and combine them for more complex use cases. Some examples of valid Outlines output types:

```python
from typing import Dict, List, Optional, Tuple, Union

output_type = float
output_type = Dict[int, str]
output_type = Tuple[Union[str, Dict[int, float]], List[bool], Optional[int]]
```

### Multiple Choice

You can use your model to do classification by using the `Literal` or `Enum` output types. For instance:

```python
from typing import Literal
from enum import Enum

class PizzaOrBurger(Enum):
    pizza = "pizza
    burger = "burger

# Equivalent multiple-choice output types
output_type = Literal["pizza", "burger"]
output_type = PizzaOrBurger
```

### Structured Objects

What we call structured object here are roughly all common types that can be turned into a JSON schema. That includes:
- A Pydantic class
- A Dataclass
- A TypedDict
- A GenSon SchemaBuilder
- A Callable (the parameters are turned into the keys and the type hinting is used to define the types of the values)

For instance:

```python
from dataclasses import dataclass

@dataclass
class Character:
    name: str
    age: int

ouput_type = Character

def character(name: str, age: int):
    return None

output_type = character
```

On top of those five types that can directly be used with a model, there are also two other json schema formats for which we need to use an Outlines-specific class: JSON schema strings and dicts. As those do not have a distinct type, we must wrap them into an `outlines.types.JsonSchema` object. For instance:

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

### Regex Patterns

You can ask your model to generate text that follows a regex pattern. As the latter are expressed as simple raw string literals, we need to use an Outlines-specific class. Thus, the regex string should be be wrapped in an `outlines.types.Regex` object. For instance:

```python
from outlines.types import Regex

regex = r"[0-9]{3}"
output_type = Regex(regex)
```

The `types` module contains a few common regex patterns stored in variable you can import and directly use as output types. There are for instance the patterns for a sentence, an email address and an isbn reference. For instance:

```python
from outlines.types import sentence

print(type(sentence)) # outlines.types.dsl.Regex
print(sentence.pattern) # [A-Z].*\s*[.!?]
```

To help you create complex regex patterns yourself, you can rely on the Outlines [regex DSL]().

### Context-Free Grammars

Outlines allows you to generate text that respects the syntax of a context-free grammar. For that, you need to define a Lark grammar. As the latter is expressed as a string, we need to use an Outlines-specific class. Thus, the CFG string should be be wrapped in an `outlines.types.CFG` object. For instance:

```python
from outlines.types import CFG

grammar_string = """
    start: expr
    expr: "{" expr "}" | "[" expr "]" |
"""
output_type = CFG(grammar_string)
```

You can find a few Lark grammar examples in the [grammars module]().

## Output type availability

The output types presented above are not available for all models as some have only limited support for structured outputs. Please refer to the documentation of the specific model you wish to use to know what output types it supports.
