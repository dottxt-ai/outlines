<div align="center">

<img src="./docs/source/_static/logo.png" alt="Outlines Logo" width=300></img>

# Outlines 〰️

Fast and reliable neural text generation.

[Install](#installation) •
[Guided generation](#guided-generation) •
[Prompting primitives](#prompting) •
[Examples](#examples) •
[Stay tuned](#stay-tuned-for)

</div>

**Outlines** 〰 is a library for neural text generation. You can think of it as a
more flexible replacement for the `generate` method in the
[transformers](https://github.com/huggingface/transformers) library.

**Outlines** 〰 helps developers *guide text generation* to build robust
interfaces with external systems. Provides generation methods that
guarantee that the output will match a regular expressions, or follow
a JSON schema.

**Outlines** 〰 provides *robust prompting primitives* that separate the prompting
from the execution logic and lead to simple implementations of few-shot
generations, ReAct, meta-prompting, agents, etc.

**Outlines** 〰 is designed as a *library* that is meant to be compatible the
broader ecosystem, not to replace it. We use as few abstractions as possible,
and generation can be interleaved with control flow, conditionals, custom Python
functions and calls to other libraries.

**Outlines** 〰 is *compatible with all models*. It only interfaces with models
via the next-token logits. It can be used with API-based models as well.

## Features

- [x] 🖍️Simple and powerful prompting primitives based on the [Jinja templating engine](https://jinja.palletsprojects.com/)
- [x] 🚄 Guided generation, including multiple choice, type constraints and dynamic stopping
- [x] ⚡ Fast [regex-guided generation](#efficient-regex-guided-generation)
- [x] 🔥 Fast [JSON generation](#efficient-json-generation-following-a-pydantic-model) following a JSON schema or a Pydantic model
- [x] 🐍 Interleave completions with loops, conditionals, and custom Python functions
- [x] 💾 Caching of generations
- [x] 🤗 Integration with Hugging Face's `transformers` models

Outlines 〰 has new releases and features coming every week! Make sure to ⭐ star and 👀 watch this repository to stay up to date.

## Stay tuned for

- Context-Free Grammar guided generation ([#178](https://github.com/outlines-dev/outlines/pull/178));
- Prompt-token alignment so you don't have to think about tokenization details ([#201](https://github.com/outlines-dev/outlines/pull/201))
- An infilling DSL ([#182](https://github.com/outlines-dev/outlines/issues/182))

You can follow [@NormalComputing](https://twitter.com/NormalComputing), [@remilouf](https://twitter.com/remilouf) or [@BrandonTWillard](https://twitter.com/BrandonTWillard) for regular updates!


## Installation

**Outlines** is available on PyPi:

``` bash
pip install outlines
```

The dependencies needed to use models are not installed by default. You will need to run:

- `pip install openai` to be able to use OpenAI [models](https://platform.openai.com/docs/api-reference).
- `pip install transformers` to be able to use Hugging Face `transformers` [models](https://huggingface.co/models?pipeline_tag=text-generation).

## Guided generation

The first step towards reliability of systems that include large language models
is to ensure that there is a well-defined interface between their output and
user-defined code. **Outlines** provides ways to control the generation of
language models to make their output more predictable.

### Early stopping

You can stop the generation after a given sequence has been found:

``` python
import outlines.text.generate as generate
import outlines.models as models

model = models.transformers("gpt2")
answer = generate.continuation(model, stop=["."])("Tell me a one-sentence joke.")
```

### Multiple choices

You can reduce the completion to a choice between multiple possibilities:

``` python
import outlines.text.generate as generate
import outlines.models as models

model = models.transformers("gpt2")

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""
answer = generate.choice(model, ["Positive", "Negative"])(prompt)
```

### Type constraint

You can instruct the model to only return integers or floats:


``` python
import outlines.text.generate as generate
import outlines.models as models

model = models.transformers("gpt2")

prompt = "1+1="
answer = generate.integer(model)(prompt)

prompt = "sqrt(2)="
answer = generate.float(model)(prompt)
```

### Efficient regex-guided generation

Outlines also comes with fast regex-guided generation. In fact, the `choice`,
`integer` and `float` functions above all use regex-guided generation under the
hood:

``` python
import outlines.models as models
import outlines.text.generate as generate


model = models.transformers("gpt2-medium")

prompt = "Is 1+1=2? "
unguided = generate.continuation(model, max_tokens=30)(prompt)
guided = generate.regex(model, r"\s*([Yy]es|[Nn]o|[Nn]ever|[Aa]lways)", max_tokens=30)(
    prompt
)

print(unguided)
# Is 1+1=2?
#
# This is probably the most perplexing question.
# As I said in one of my articles describing how
# I call 2 and 1, there isn't

print(guided)
# Is 1+1=2? Always
```

``` python
import outlines.models as models
import outlines.text.generate as generate


model = models.transformers("gpt2-medium")

prompt = "What is the IP address of the Google DNS servers? "
unguided = generate.continuation(model, max_tokens=30)(prompt)
guided = generate.regex(
    model,
    r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
    max_tokens=30,
)(prompt)

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

import outlines.models as models
import outlines.text.generate as generate

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


model = models.transformers("gpt2", device="cuda")

# Construct guided sequence generator
generator = generate.json(model, Character, max_tokens=100)

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

parsed = Character.model_validate_json(sequence)
print(parsed)
# name='piggyback' age=23 armor=<Armor.chainmail: 'chainmail'> weapon=<Weapon.sword: 'sword'> strength=0
```

The method works with union types, optional types, arrays, nested schemas, etc. Some field constraints are [not supported yet](https://github.com/outlines-dev/outlines/issues/215), but everything else should work.

## Prompting

Writing prompts by concatenating strings in pure Python quickly becomes
cumbersome: the prompt building logic gets entangled with the rest of the
program, and the structure of the rendered prompt is obfuscated.**Outlines**
makes it easier to write and manage prompts by encapsulating templates inside
"template functions".

These functions make it possible to neatly separate the prompt logic from the
general program logic; they can be imported from other modules and libraries.

Template functions require no superfluous abstraction, they use the Jinja2
templating engine to help build complex prompts in a concise manner:

``` python
import outlines.text as text
import outlines.models as models


examples = [
    ("The food was digusting", "Negative"),
    ("We had a fantastic night", "Positive"),
    ("Recommended", "Positive"),
    ("The waiter was rude", "Negative")
]

@text.prompt
def labelling(to_label, examples):
    """You are a sentiment-labelling assistant.

    {% for example in examples %}
    {{ example[0] }} // {{ example[1] }}
    {% endfor %}
    {{ to_label }} //
    """

model = models.transformers("gpt2")
prompt = labelling("Just awesome", examples)
answer = text.generate.continuation(model, max_tokens=100)(prompt)
```

### Tools

We can teach language models to call external functions to get additional
informations or perform tasks, by encoding the functions' description in the
prompt. To avoid duplicating information between the function definition and the
description passed to the prompt, we define custom Jinja filters that can
extract the function's name, description, signature and source:


``` python
from typing import Callable, List
import outlines.text as text


def google_search(query: str):
    """Google Search"""
    pass


def wikipedia_search(query: str):
    """Wikipedia Search"""
    pass


@text.prompt
def my_commands(tools: List[Callable]):
    """AVAILABLE COMMANDS:

    {% for tool in tools %}
    TOOL
    {{ tool | name }}, {{ tool | description }}, args: {{ tool | signature }}
    {{ tool | source }}
    {% endfor %}
    """


prompt = my_commands([google_search, wikipedia_search])
```

### Response models

We can instruct models to return their output in a pre-defined format, often
JSON. To avoid duplicating information between the function definition and the
description passed to the prompt we define a custom Jinja filter that can
extract the expected response's schema:

``` python
from pydantic import BaseModel
import outlines.text as text


class Joke(BaseModel):
    joke: str
    explanation: str


@text.prompt
def joke_ppt(response_model):
    """Tell a joke and explain why the joke is funny.

    RESPONSE FORMAT:
    {{ response_model | schema }}
    """


joke_ppt(Joke)
# Tell a joke and explain why the joke is funny.
#
# RESPONSE FORMAT:
# {
#    "joke": "The joke"
#    "explanation": "The explanation of why the joke is funny"
#  }
```

With these prompting primitives **Outlines** makes building agents like
[AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT),
[BabyAGI](https://github.com/yoheinakajima/babyagi),
[ViperGPT](https://viper.cs.columbia.edu/) or [Transformers
Agent](https://huggingface.co/docs/transformers/transformers_agents) easier by
removing boilerplate prompting code.

## Contributing

### What contributions?

We currently only accept bug fixes and documentation contributions. If you have a
feature request, please start a new
[discussion](https://github.com/outlines-dev/outlines/discussions). The
issue tracker is only intended for actionable items.

### How to contribute?

Run `pip install -e .[test]` or `conda env create -f environment.yml`. To build the documentation you will also need to run `pip install -r requirements-doc.txt`.

Before pushing your code to repository please run `pre-commit run --all-files` and `pytest` to make sure that the code is formatted correctly and that the tests pass.

Do not hesitate to open a draft PR before your contribution is ready, especially if you have questions and/or need feedback.

## Examples

- [Pick the odd one out](https://github.com/outlines-dev/outlines/blob/main/examples/pick_odd_one_out.py)
- [Meta prompting](https://github.com/outlines-dev/outlines/blob/main/examples/meta_prompting.py)
- [ReAct](https://github.com/outlines-dev/outlines/blob/main/examples/react.py)
- [Generate code to solve math problems](https://github.com/outlines-dev/outlines/blob/main/examples/math_generate_code.py)
- [BabyAGI](https://github.com/outlines-dev/outlines/blob/main/examples/babyagi.py)
- [Uncertainty](https://github.com/outlines-dev/outlines/blob/main/examples/sampling.ipynb)
- [Simulation-based inference](https://github.com/outlines-dev/outlines/blob/main/examples/simulation_based_inference.ipynb)


## Cite Outlines

```
@article{willard2023efficient,
  title={Efficient Guided Generation for LLMs},
  author={Willard, Brandon T and Louf, R{\'e}mi},
  journal={arXiv preprint arXiv:2307.09702},
  year={2023}
}
```

## License

Outlines is open-source and licensed under the [Apache License 2.0](LICENSE).
