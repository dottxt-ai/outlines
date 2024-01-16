<div align="center">

# Outlines 〰️

<img src="./docs/assets/images/logo.png" alt="Outlines Logo" width=300></img>

[![Pypi][pypi-badge]][pypi]
[![Contributors][contributors-badge]][contributors]
[![Discord][discord-badge]][discord]
[![Twitter][twitter-badge]][twitter]

*Robust (guided) text generation.*

</div>

Made with ❤👷️ by the team at [.txt](https://dottxt.co).
We'd love to have your [feedback][discord]!

``` bash
pip install outlines
```

First time here? Go to our [setup guide](https://outlines-dev.github.io/outlines/welcome)

## Features

- [x] 🤖 [Multiple model integrations](https://outlines-dev.github.io/outlines/installation): OpenAI, transformers, llama.cpp, exllama2, mamba
- [x] 🖍️ Simple and powerful prompting primitives based on the [Jinja templating engine](https://jinja.palletsprojects.com/)
- [x] 🚄 [Multiple choices](#multiple-choices), [type constraints](#type-constraint) and dynamic stopping
- [x] ⚡ Fast [regex-guided generation](#efficient-regex-guided-generation)
- [x] 🔥 Fast [JSON generation](#efficient-json-generation-following-a-pydantic-model) following a JSON schema or a Pydantic model
- [x] 📝 [Grammar-guided generation](#using-context-free-grammars-to-guide-generation)
- [x] 🐍 Interleave completions with loops, conditionals, and custom Python functions
- [x] 💾 Caching of generations
- [x] 🗂️ Batch inference
- [x] 🚀 [Serve with vLLM](https://outlines-dev.github.io/outlines/reference/vllm)


Outlines 〰 has new releases and features coming every week. Make sure to ⭐ star and 👀 watch this repository, follow [@dottxtai][twitter] to stay up to date!


## Guided generation

The first step towards reliability of systems that include large language models
is to ensure that there is a well-defined interface between their output and
user-defined code. **Outlines** provides ways to control the generation of
language models to make their output more predictable.

### Multiple choices

You can reduce the completion to a choice between multiple possibilities:

``` python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
```

### Type constraint

You can instruct the model to only return integers or floats:


``` python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

prompt = "1+1="
answer = outlines.generate.format(model, int)(prompt)

prompt = "sqrt(2)="

generator = outlines.generate.format(model, float)
answer = generator(prompt)
```

### Efficient regex-guided generation

Outlines also comes with fast regex-guided generation. In fact, the `choice` and
`format` functions above all use regex-guided generation under the
hood:

``` python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

prompt = "What is the IP address of the Google DNS servers? "

generator = outlines.generate.text(model)
unguided = generator(prompt, max_tokens=30)

generator = outlines.generate.regex(
    model,
    r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
)
guided = generator(prompt, max_tokens=30)

print(unguided)
# What is the IP address of the Google DNS servers?
#
# Passive DNS servers are at DNS servers that are private.
# In other words, both IP servers are private. The database
# does not contain Chelsea Manning

print(guided)
# What is the IP address of the Google DNS servers?
# 2.2.6.1
```

Unlike other libraries, regex-guided generation in Outlines is almost as fast
as non-guided generation.

### Efficient JSON generation following a Pydantic model

Outlines 〰 allows to guide the generation process so the output is *guaranteed* to follow a [JSON schema](https://json-schema.org/) or [Pydantic model](https://docs.pydantic.dev/latest/):

```python
from enum import Enum
from pydantic import BaseModel, constr

import outlines
import torch


class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    weapon: Weapon
    strength: int


model = outlines.models.transformers("mistralai/Mistral-7B-v0.1", device="cuda")

# Construct guided sequence generator
generator = outlines.generate.json(model, Character, max_tokens=100)

# Draw a sample
rng = torch.Generator(device="cuda")
rng.manual_seed(789001)

sequence = generator("Give me a character description", rng=rng)
print(sequence)
# {
#   "name": "clerame",
#   "age": 7,
#   "armor": "plate",
#   "weapon": "mace",
#   "strength": 4171
# }

sequence = generator("Give me an interesting character description", rng=rng)
print(sequence)
# {
#   "name": "piggyback",
#   "age": 23,
#   "armor": "chainmail",
#   "weapon": "sword",
#   "strength": 0
# }
```

The method works with union types, optional types, arrays, nested schemas, etc. Some field constraints are [not supported yet](https://github.com/outlines-dev/outlines/issues/215), but everything else should work.

### Efficient JSON generation following a JSON Schema

Sometimes you just want to be able to pass a JSON Schema instead of a Pydantic model. We've got you covered:

``` python
import outlines

schema = '''{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "maxLength": 10,
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}'''

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1", device="cuda")
generator = outlines.generate.json(model, schema)
sequence = generator("Give me a character description")
```

### Using context-free grammars to guide generation

Formal grammars rule the world, and Outlines makes them rule LLMs too. You can pass any context-free grammar in the EBNF format and Outlines will generate an output that is valid to this grammar:

``` python
import outlines

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

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1", device="cuda")
generator = outlines.generate.cfg(model, arithmetic_grammar)
sequence = generator("Write a formula that returns 5 using only additions and subtractions.")

# It looks like Mistral is not very good at arithmetics :)

print(sequence)
# 1+3-2-4+5-7+8-6+9-6+4-2+3+5-1+1
```

This was a very simple grammar, and you can use `outlines.generate.cfg` to generate syntactically valid Python, SQL, and much more than this. Any kind of structured text, really. All you have to do is search for "X EBNF grammar" on the web, and take a look at the [Outlines Grammars repository](https://github.com/outlines-dev/grammars).

### Open functions

Outlines can infer the structure of the output from the signature of a function. The result is a dictionary, and can be passed directly to the function using the usual dictionary expansion syntax `**`:

```python
import outlines


def add(a: int, b: int):
    return a + b

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")
generator = outlines.generate.json(model, add)
result = generator("Return two integers named a and b respectively. a is odd and b even.")

print(add(**result))
# 3
```

A great advantage of passing functions directly to specify the structure is that the structure of the LLM will change with the function's definition. No need to change the code at several places!

## Prompting

Building prompts can get messy. **Outlines** makes it easier to write and manage
prompts by encapsulating templates inside "template functions".

These functions make it possible to neatly separate the prompt logic from the
general program logic; they can be imported from other modules and libraries.

Template functions require no superfluous abstraction, they use the Jinja2
templating engine to help build complex prompts in a concise manner:

``` python
import outlines

examples = [
    ("The food was disgusting", "Negative"),
    ("We had a fantastic night", "Positive"),
    ("Recommended", "Positive"),
    ("The waiter was rude", "Negative")
]

@outlines.prompt
def labelling(to_label, examples):
    """You are a sentiment-labelling assistant.

    {% for example in examples %}
    {{ example[0] }} // {{ example[1] }}
    {% endfor %}
    {{ to_label }} //
    """

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")
prompt = labelling("Just awesome", examples)
answer = outlines.generate.text(model)(prompt, max_tokens=100)
```

## Join us

- 💡 **Have an idea?** Come chat with us on [Discord][discord]
- 🔨 **Want to contribute?** Consult our [contribution guide](https://outlines-dev.github.io/outlines/community/contribute/).
- 🐞 **Found a bug?** Open an [issue](https://github.com/outlines-dev/outlines/issues)



[contributors]: https://github.com/outlines-dev/outlines/graphs/contributors
[contributors-badge]: https://img.shields.io/github/contributors/outlines-dev/outlines?style=flat-square&logo=github&logoColor=white&color=ECEFF4
[twitter]: https://twitter.com/dottxtai
[discord]: https://discord.gg/R9DSu34mGd
[discord-badge]: https://img.shields.io/discord/1182316225284554793?color=81A1C1&logo=discord&logoColor=white&style=flat-square
[twitter-badge]: https://img.shields.io/twitter/follow/dottxtai?style=social
[pypi]: https://pypi.org/project/outlines/
[pypi-badge]: https://img.shields.io/pypi/v/outlines?color=ECEFF4&logo=python&logoColor=white&style=flat-square
