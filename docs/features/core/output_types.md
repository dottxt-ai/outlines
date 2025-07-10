---
title: Output Types
---

# Output Types

Outlines provides a simple and intuitive way of defining the output structure of text generation. Possible output formats include basic Python types, multiple-choices, JSON schemas, regular expressions and context-free grammars.

## Overview

Outlines models accept a __prompt__ and an __output type__ when they are invoked, as well as additional inference keyword arguments that are forwarded on to the underlying model.

Output types can be from the general Python ecosystem, including:
- Most native Python types, such as `int` or `str`
- Types from the `typing` module, such as `Literal`, `List`, `Dict`, `Enum`, etc
- Types from popular third party libraries such as Pydantic or GenSON.

Outlines also provides special classes for certain output structures (more details below):
- Multiple choices with `Choice`
- JSON schemas with `JsonSchema`
- Regular expressions with `Regex`
- Context-free grammars with `CFG`

The general idea is that you should provide as an output type what you would give as the type hint of the return type of a function.

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

With an Outlines model, you can generate text that respects the type hints above by providing those as the output type:

```python
model("How many minutes are there in one hour", int) # "60"
model("Pizza or burger", Literal["pizza", "burger"]) # "pizza"
model("Create a character", Character, max_new_tokens=100) # '{"name": "James", "birth_date": "1980-05-10)", "skills": ["archery", "negotiation"]}'
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

### Basic Python Types

The most straightforward form of structured generation is to return an answer that conforms to a given basic type such as an int or a python list. You can use the basic Python types and the types from the `typing` library. For instance:

```python
from typing import Dict

output_type = float # example of valid value: "0.05"
output_type = bool # example of valid value: "True"
output_type = Dict[int, str] # example of valid value: "{1: 'hello', 2: 'there'}"
```

You can combine types to create more complex response formats by relying on collection types and types such as `Union` and `Optional`. Let's consider for instance the output type below used to represent semi-structured data:

```python
from typing import Dict, List, Optional, Tuple, Union

output_type = Dict[str, Union[int, str, List[Tuple[str, Optional[float]]]]]
```

Values created with this output type would be dictionaries with string as keys and values made of either an integer, a string or a list of two elements tuples: a string and either a float or None. Example of a valid response for text generated with this output type (it would be contained in a string):

```json
{
    "name": "Alice",
    "age": 30,
    "metrics": [("engagement", 0.85), ("satisfaction", None)]
}
```

### Multiple Choices

Outlines supports multiple choice classification by using the `Literal` or `Enum` output types. For instance:

```python
from enum import Enum
from typing import Literal

class PizzaOrBurger(Enum):
    pizza = "pizza"
    burger = "burger"

# Equivalent multiple-choice output types
output_type = Literal["pizza", "burger"]
output_type = PizzaOrBurger
```

Additionally, you can use the Outlines-specific type `Choice` that takes a `list` as an argument. This type is useful in situations in which the list of choices is dynamic.

For instance:

```python
from outlines.types import Choice

def get_multiple_choices() -> list:
    # we could have something complex here
    return ["pizza", "burger"]

output_type = Choice(get_multiple_choices())
```

### JSON Schemas

Multiple different common Python types are often used to store information equivalent to a JSON schema. The following can be used in Outlines to generate text that respects a JSON schema:

- A Pydantic class
- A Dataclass
- A TypedDict
- A [GenSON](https://github.com/wolverdude/GenSON) `SchemaBuilder`
- A Callable (the parameters are turned into the keys and the type hinting is used to define the types of the values)

For instance:

```python
from dataclasses import dataclass

@dataclass
class Character:
    name: str
    age: int

output_type = Character

def character(name: str, age: int):
    return None

output_type = character
```

There are two other JSON schema formats that require Outlines-specific classes: JSON schema strings and dictionaries.

As those are contained in regular Python strings or dictionaries, the associated output format would be ambiguous if they were to be provided directly. As a result, Outlines requires them to be wrapped in a `outlines.types.JsonSchema` object. For instance:

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

`JsonSchema` accepts two optional parameters:
- `whitespace_pattern` (defaults to `None`): specifies the pattern to use for JSON syntactic whitespace. If none is provided, the default permissive JSON whitespace rules are used.
- `ensure_ascii` (defaults to `True`): defines the value to use for the argument `ensure_ascii` of the `json.dumps` method. If false, non-ASCII characters will be turned into unicodes.

### Regex Patterns

Outlines provides support for text generation constrained by regular expressions. Since regular expressions are expressed as simple raw string literals, regex strings must wrapped in an `outlines.types.Regex` object.

```python
from outlines.types import Regex

regex = r"[0-9]{3}"
output_type = Regex(regex)
```

The `outlines.types` module contains a few common regex patterns stored in variables you can import and directly use as output types. Common patterns include a sentence, an email address and an [ISBN reference](https://en.wikipedia.org/wiki/ISBN). For instance:

```python
from outlines.types import sentence

print(type(sentence)) # outlines.types.dsl.Regex
print(sentence.pattern) # [A-Z].*\s*[.!?]
```

To help you create complex regex patterns yourself, you can use the Outlines [regex DSL](../../utility/regex_dsl).

### Context-Free Grammars

Outlines allows you to generate text that respects the syntax of a context-free grammar. Context-free grammars are defined using [Lark](https://lark-parser.readthedocs.io/en/latest/index.html), a grammar language. Since grammars are expressed as a string, Large CFG strings should be be wrapped in an `outlines.types.CFG` object. For instance:

```python
from outlines.types import CFG

grammar_string = """
    start: expr
    expr: "{" expr "}" | "[" expr "]" |
"""
output_type = CFG(grammar_string)
```

You can find a few Lark grammar examples in the [grammars module](../../api_reference/grammars.md).

## Output type availability

The output types presented above are not available for all models as some have only limited support for structured outputs. Please refer to the documentation of the specific model you wish to use to know what output types it supports.
