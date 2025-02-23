<div align="center" style="margin-bottom: 1em;">

<img src="./docs/assets/images/logo.png" alt="Outlines Logo" width=500></img>


 🗒️ *Make LLMs speak the language of every application.* 🗒️

Made with ❤👷️ by the team at [.txt](https://dottxt.co).

[![Documentation][documentation-badge]][documentation]
[![Contributors][contributors-badge]][contributors]
[![Downloads][downloads-badge]][pypistats]
[![Discord][discord-badge]][discord]

[Youtube channel][youtube-dottxt] | [.txt blog][blog-dottxt] | [Twitter][dottxt-twitter]


</div>


``` bash
pip install outlines
```

First time here? Go to our [setup guide](https://dottxt-ai.github.io/outlines/latest/welcome/)

## Features

- 🤖 [Multiple model integrations](https://dottxt-ai.github.io/outlines/latest/installation): OpenAI, transformers, llama.cpp, mamba
- 🔥 Fast [JSON generation](#efficient-json-generation-following-a-pydantic-model) following a JSON schema or a Pydantic model
- 🚄 [Multiple choices](#multiple-choices), [type constraints](#type-constraint) and dynamic stopping
- 📝 Generate text that follows a [regex](#efficient-regex-structured-generation) or a [context-free grammar](#using-context-free-grammars-to-guide-generation)
- 🖍️ Simple and powerful prompting primitives based on the [Jinja templating engine](https://jinja.palletsprojects.com/)
- 🚀 [Serve with vLLM](https://dottxt-ai.github.io/outlines/latest/reference/serve/vllm), with official Docker image, [`outlinesdev/outlines`](https://hub.docker.com/r/outlinesdev/outlines)!


Outlines  has new releases and features coming every week. Make sure to ⭐ star and 👀 watch this repository, follow [@dottxtai][dottxt-twitter] to stay up to date!

## Why should I use structured generation?

* It doesn't add any overhead during inference (cost-free)
* It allows Open Source models to beat closed source models ([Mistral](https://x.com/dottxtai/status/1797692104023363765), [GPT-4](https://x.com/dottxtai/status/1798443290913853770))
* [It speeds up inference](http://blog.dottxt.co/coalescence.html)
* [It improves the performance of base models (GSM8K)](http://blog.dottxt.co/performance-gsm8k.html)
* [It improves the performance of finetuned models (CoNNL)](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora)
* [It improves model efficiency (less examples needed)](https://huggingface.co/blog/evaluation-structured-outputs)

## .txt company

<div align="center">
<img src="./docs/assets/images/dottxt.png" alt="Outlines Logo" width=100></img>
</div>

We started a company to keep pushing the boundaries of structured generation. Learn more about [.txt](https://twitter.com/dottxtai), and  [give our .json API a try](https://h1xbpbfsf0w.typeform.com/to/ZgBCvJHF) if you need a hosted solution ✨

## Structured generation

The first step towards reliability of systems that include large language models
is to ensure that there is a well-defined interface between their output and
user-defined code. **Outlines** provides ways to control the generation of
language models to make their output more predictable.

The following methods of structured generation are supported:

- [Multiple choices](#multiple-choices)
- [Type constraints](#type-constraint)
- [Efficient regex-structured generation](#efficient-regex-structured-generation)
- [Efficient JSON generation following a Pydantic model](#efficient-json-generation-following-a-pydantic-model)
- [Using context-free grammars to guide generation](#using-context-free-grammars-to-guide-generation)
- [Open functions](#open-functions)

### Chat template tokens

Outlines does not manage chat templating tokens when using instruct models. You must apply the chat template tokens to the prompt yourself. Chat template tokens are not needed for base models.

Please see [the documentation](https://dottxt-ai.github.io/outlines/latest/reference/chat_templating) on chat templating for more.

### Multiple choices

You can reduce the completion to a choice between multiple possibilities:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

# You must apply the chat template tokens to the prompt!
# See below for an example.
prompt = """
<|im_start|>system
You extract information from text.
<|im_end|>

<|im_start|>user
What food does the following text describe?

Text: I really really really want pizza.
<|im_end|>
<|im_start|>assistant
"""

answer = model(prompt, outlines.Choice(["Pizza", "Pasta", "Salad", "Dessert"]))
# Likely answer: Pizza
```

You can also pass in choices with an `Enum`:

````python
from enum import Enum

class Food(str, Enum):
    pizza = "Pizza"
    pasta = "Pasta"
    salad = "Salad"
    dessert = "Dessert"

answer = model(prompt, outlines.Choice(Food))
# Likely answer: Pizza
````

### Type constraints

You can instruct the model to only return integers or floats:


``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines

model_name = "WizardLM/WizardMath-7B-V1.1"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

prompt = "<s>result of 9 + 9 = 18</s><s>result of 1 + 2 = "
answer = outlines.generate.format(model, int)(prompt)
print(answer)
# 3

prompt = "sqrt(2)="
answer = model(prompt, outlines.types.number, max_tokens=10)
print(answer)
# 1.41421356
```

### Efficient regex-structured generation

Outlines also comes with fast regex-structured generation. In fact, the `choice` and
`format` functions above all use regex-structured generation under the
hood:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
from outlines import regex

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

prompt = """
<|im_start|>system You are a helpful assistant.
<|im_end|>

<|im_start|>user
What is an IP address of the Google DNS servers?
<|im_end|>
<|im_start|>assistant
The IP address of a Google DNS server is

"""

unstructured = model(prompt, max_tokens=10)

structured = model(
    prompt,
    outlines.Regex(r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"),
    max_tokens=30
)

print(unstructured)
# 8.8.8.8
#
# <|im_end|>

print(structured)
# 8.8.8.8
```

Unlike other libraries, regex-structured generation in Outlines is almost as fast
as non-structured generation.

### Efficient JSON generation following a Pydantic model

Outlines users can guide the generation process so the output is *guaranteed* to follow a [JSON schema](https://json-schema.org/) or [Pydantic model](https://docs.pydantic.dev/latest/):

```python
from enum import Enum
from pydantic import BaseModel, constr
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

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


model_name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

# Construct structured sequence generator
generator = outlines.Generator(model, Character)

# Draw a sample
seed = 789001
character = generator("Give me a character description", seed=seed)

print(repr(character))
# Character(name='Anderson', age=28, armor=<Armor.chainmail: 'chainmail'>, weapon=<Weapon.sword: 'sword'>, strength=8)

prompt = "Give me an interesting character description"
character = model(prompt, outlines.JsonType(Character), seed=seed)

print(repr(character))
# Character(name='Vivian Thr', age=44, armor=<Armor.plate: 'plate'>, weapon=<Weapon.crossbow: 'crossbow'>, strength=125)
```

The method works with union types, optional types, arrays, nested schemas, etc. Some field constraints are [not supported yet](https://github.com/dottxt-ai/outlines/issues/215), but everything else should work.

### Efficient JSON generation following a JSON Schema

Sometimes you just want to be able to pass a JSON Schema instead of a Pydantic model. We've got you covered:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

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

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

prompt = "Give me a character description"
character = model(prompt, outlines.JsonType(schema))
```

### Using context-free grammars to guide generation

Formal grammars rule the world, and Outlines makes them rule LLMs too. You can pass any context-free grammar in the EBNF format and Outlines will generate an output that is valid to this grammar:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

arithmetic_grammar = """
    ?start: expression

    ?expression: term (("+" | "-") term)*

    ?term: factor (("*" | "/") factor)*

    ?factor: NUMBER
           | "-" factor
           | "(" expression ")"

    %import common.NUMBER
"""

model_name = "WizardLM/WizardMath-7B-V1.1"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)

prompt = "Alice had 4 apples and Bob ate 2. Write an expression for Alice's apples:"
sequence = model(prompt, outlines.types.Cfg(arithmetic_grammar))

print(sequence)
# (8-2)
```

This was a very simple grammar, and you can use `outlines.generate.cfg` to generate syntactically valid Python, SQL, and much more than this. Any kind of structured text, really. All you have to do is search for "X EBNF grammar" on the web, and take a look at the [Outlines `grammars` module](https://github.com/dottxt-ai/outlines/tree/main/outlines/grammars).

### Open functions

Outlines can infer the structure of the output from the signature of a function. The result is a dictionary, and can be passed directly to the function using the usual dictionary expansion syntax `**`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines


def add(a: int, b: int):
    return a + b

model_name = "WizardLM/WizardMath-7B-V1.1"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)
result = model(
    "Return json with two integers named a and b respectively. a is odd and b even.",
    add
)

print(add(**result))
# 3
```

A great advantage of passing functions directly to specify the structure is that the structure of the LLM will change with the function's definition. No need to change the code at several places!

You can also embed various functions into an enum to generate params:

```python
from enum import Enum
from functools import partial

from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines


def add(a: int, b: int) -> int:
    return a + b

def mul(c: float, d: float) -> float:
    return c * d

class Operation(Enum):
    add = partial(add)
    mul = partial(mul)

model_name = "WizardLM/WizardMath-7B-V1.1"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_name),
    AutoTokenizer.from_pretrained(model_name)
)
generator = outlines.Generator(model, outlines.types.JsonType(Operation))
result = generator("Return json with two float named c and d respectively. c is negative and d greater than 1.0.")
#print(result)
# {'c': -3.14, 'd': 1.5}
```

## Prompting

Building prompts can get messy. **Outlines** makes it easier to write and manage
prompts by encapsulating templates inside "template functions". Template
functions use the Jinja2 templating engine to help build complex prompts in a
concise manner.

Template functions are created by loading a Jinja2 template from a text file.
Assume you have the following prompt template defined in `prompt.txt`:

``` text
You are a sentiment-labelling assistant.

{% for example in examples %}
{{ example[0] }} // {{ example[1] }}
{% endfor %}
{{ to_label }} //
```

You can then load it and call it with:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines

examples = [
    ("The food was disgusting", "Negative"),
    ("We had a fantastic night", "Positive"),
    ("Recommended", "Positive"),
    ("The waiter was rude", "Negative")
]

labelling = outlines.Template.from_file("prompt.txt")
prompt = labelling("Just awesome", examples)
```

This helps:

- Keep content separate from the code
- Design "white space perfect" prompts

It is more maintainable and means prompts can be versioned separately from the code.

## Join us

- 💡 **Have an idea?** Come chat with us on [Discord][discord]
- 🔨 **Want to contribute?** Consult our [contribution guide](https://dottxt-ai.github.io/outlines/latest/community/contribute/).
- 🐞 **Found a bug?** Open an [issue](https://github.com/dottxt-ai/outlines/issues)


## Cite Outlines

```
@article{willard2023efficient,
  title={Efficient Guided Generation for LLMs},
  author={Willard, Brandon T and Louf, R{\'e}mi},
  journal={arXiv preprint arXiv:2307.09702},
  year={2023}
}
```

[documentation]: https://dottxt-ai.github.io/outlines/latest/welcome/
[documentation-badge]: https://img.shields.io/readthedocs/outlines
[contributors]: https://github.com/dottxt-ai/outlines/graphs/contributors
[contributors-badge]: https://img.shields.io/github/contributors/dottxt-ai/outlines?style=flat-square&logo=github&logoColor=white&color=ECEFF4
[dottxt-twitter]: https://twitter.com/dottxtai
[discord]: https://discord.gg/R9DSu34mGd
[discord-badge]: https://img.shields.io/discord/1182316225284554793?color=81A1C1&logo=discord&logoColor=white&style=flat-square
[downloads-badge]: https://img.shields.io/pypi/dm/outlines?color=89AC6B&logo=python&logoColor=white&style=flat-square
[pypistats]: https://pypistats.org/packages/outlines
[dottxt-twitter-badge]: https://img.shields.io/twitter/follow/dottxtai?style=social
[youtube-dottxt]: https://www.youtube.com/@dottxt-ai
[blog-dottxt]: https://blog.dottxt.co/
