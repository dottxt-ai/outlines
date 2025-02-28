---
title: Generation
---

# Generation

Once an [Outlines model](../models/models.md) is constructed you can use a `Generator` to generate text. The second argument to the `Generator` constructor is the output type. Standard LLM generation is possible by not passing any output type. Otherwise, you can provide a variety of different output types to benefit from structured generation. (For a detailed technical explanation of how structured generation works, you may review the [Structured Generation Explanation](./structured_generation_explanation.md) page)

Before generating text, you must construct an `outlines.model`. Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)
```

### Text generator

```python
generator = outlines.Generator(model)

result = generator("Question: What's 2+2? Answer:", max_tokens=100)
print(result)
# The answer is 4
```

### [Multi-label classification](./choices.md)

Outlines allows you to do multi-label classification by guiding the model so it can only output either of the specified choices:

```python
import outlines
from outlines.types import Choice

generator = outlines.Generator(model, Choice(["Blue", "Red", "Yellow"]))

color = generator("What is the closest color to Indigo? ")
print(color)
# Blue
```

### [JSON-structured generation](./json.md)

Outlines can guide models so that they output valid JSON **100%** of the time. You can either specify the structure using [Pydantic][pydantic]{:target="_blank"} or a string that contains a [JSON Schema][jsonschema]{:target="_blank"}:

=== "Pydantic"

    ```python
    from enum import Enum
    from pydantic import BaseModel, constr, conint

    from outlines.types import JsonType
    import outlines

    class Armor(str, Enum):
        leather = "leather"
        chainmail = "chainmail"
        plate = "plate"


    class Character(BaseModel):
        name: constr(max_length=10)
        age: conint(gt=18, lt=99)
        armor: Armor
        strength: conint(gt=1, lt=100)

    generator = outlines.Generator(model, JsonType(Character))

    character = generator(
        "Generate a new character for my awesome game: "
        + "name, age (between 1 and 99), armor and strength. "
        )
    print(character)
    # name='Orla' age=21 armor=<Armor.plate: 'plate'> strength=8
    ```

=== "JSON Schema"

    ```python
    import outlines
    from outlines.types import JsonType

    schema = """{
        "$defs": {
            "Armor": {
                "enum": ["leather", "chainmail", "plate"],
                "title": "Armor",
                "type": "string"
            }
        },
        "properties": {
            "name": {"maxLength": 10, "title": "Name", "type": "string"},
            "age": {"title": "Age", "type": "integer"},
            "armor": {"$ref": "#/$defs/Armor"},
            "strength": {"title": "Strength", "type": "integer"}\
        },
        "required": ["name", "age", "armor", "strength"],
        "title": "Character",
        "type": "object"
    }"""

    generator = outlines.Generator(model, JsonType(schema))
    character = generator(
        "Generate a new character for my awesome game: "
        + "name, age (between 1 and 99), armor and strength. "
        )
    print(character)
    # {'name': 'Yuki', 'age': 24, 'armor': 'plate', 'strength': 3}
    ```

!!! Note

    We advise you to constrain the length of the strings fields when first testing your schema, especially with small models.

### [Grammar-structured generation](./cfg.md)

Outlines also allows to generate text that is valid to any [context-free grammar][cfg]{:target="_blank"} (CFG) in the [EBNF format][ebnf]{:target="_blank"}. Grammars can be intimidating, but they are a very powerful tool! Indeed, they determine the syntax of every programming language, valid chess moves, molecule structure, can help with procedural graphics generation, etc.

Here we show a simple example of a grammar that defines arithmetic operations:

```python
import outlines
from outlines.types import CFG

arithmetic_grammar = """
    ?start: sum

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: atom
        | product "*" atom  -> mul
        | product "/" atom  -> div

    ?atom: NUMBER           -> number
         | "-" atom         -> neg
         | "(" sum ")"

    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""

generator = outlines.Generator(model, CFG(arithmetic_grammar, max_tokens=100))

result = generator("Question: How can you write 5*5 using addition?\nAnswer:")
print(result)
# 5+5+5+5+5
```


EBNF grammars can be cumbersome to write. This is why Outlines provides grammar definitions in the `outlines.grammars.` module

```python
import outlines
from outlines import grammars
from outlines.types import CFG

generator = outlines.Generator(model, CFG(grammars.arithmetic, max_tokens=100))

result = generator("Question: How can you write 5*5 using addition?\nAnswer:")
print(result)
# 5+5+5+5+5
```

The available grammars are listed [here](https://github.com/dottxt-ai/outlines/tree/main/outlines/grammars).


### [Regex-structured generation](./regex.md)

Slightly simpler, but no less useful, Outlines can generate text that is in the language of a [regular expression](https://www.regular-expressions.info/tutorial.html). For instance to force the model to generate IP addresses:

```python
import outlines
from outlines.types import Regex

regex_str = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
generator = outlines.Generator(model, Regex(regex_str))

result = generator("What is the IP address of localhost?\nIP: ")
print(result)
# 127.0.0.100
```

[jsonschema]: https://json-schema.org/learn/getting-started-step-by-step
[pydantic]: https://docs.pydantic.dev/latest
[cfg]: https://en.wikipedia.org/wiki/Context-free_grammar
[ebnf]: https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
