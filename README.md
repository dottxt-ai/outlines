<div align="center">

<img src="./docs/source/_static/logo.png" alt="Outlines Logo" width=300></img>

# Outlines

Build _reliable_ workflows based on interactions with generative models.

[Prompting](#prompting) •
[Controlled generation](#controlled-generation) •
[Agents](#agents-example) •
[Sampling](#sampling-uncertainty-simulation-based-inference) •
[Parallel execution](#vectorization-and-parallel-execution) •
[Examples](#examples)

</div>

**Outlines** allows you to control and diagnose interactions with LLMs more effectively. Modern language models are powerful and versatile, but the way they interface with existing systems [can be very brittle](https://github.com/Significant-Gravitas/Auto-GPT/labels/invalid_json), their outputs [can be unreliable](https://arxiv.org/abs/2302.04023), and complex workflows (agents) can introduce a lot of error-prone code duplication. Outlines provides robust prompting primitives that separate the prompting from the execution logic and lead to simple implementations of few-shot generations, ReAct, meta-prompting, agents, etc. Outlines helps developers control text generation and produce predictable outputs that make the interaction with user code more robust. Its sampling-first approach allows one to diagnose issues with model-generated output more easily, and implement more robust generation methods such as [self-consistency](https://arxiv.org/abs/2203.11171) or [DiVeRSe](https://arxiv.org/abs/2206.02336).

**Outlines** is designed as a library that integrates well with the broader Python environment. Generation can be interleaved with control flow or custom function calls, prompts can be imported from other modules or libraries.


## Features

- [x] Simple and powerful prompting primitives based on the [Jinja templating engine](https://jinja.palletsprojects.com/).
- [x] Interleave completions with loops, conditionals, and custom Python functions
- [x] Caching of generations
- [x] Integration with OpenAI and HuggingFace models
- [x] Controlled generation, including multiple choice, type constraints and dynamic stopping
- [x] Sampling of multiple sequences
- [x] Vectorized execution


## Installation

**Outlines** is available on PyPi:

``` bash
pip install outlines
```

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

complete = models.text_completion.openai("text-davinci-003")
prompt = labelling("Just awesome", examples)
answer = complete(prompt)
```

## Chaining with loops and conditionals ([example](https://github.com/normal-computing/outlines/blob/readme/examples/react.py))

**Outlines** comes with very few abstractions, and is designed to blend into existing code and integrate with the rest of the ecosystem.

``` python
reviews = ["Just awesome", "Avoid", "Will come back"]

def send_notification(review):
    """This function sends a notification with the review's content."""
    ...

for review in reviews:
    prompt = labelling(review, examples)
    answer = model(prompt)
    if answer == "Positive":
        send_notification(review)
```

## Agents ([example](https://github.com/normal-computing/outlines/blob/readme/examples/babyagi.py))

**Outlines** makes building agents like [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT), [BabyAGI](https://github.com/yoheinakajima/babyagi), [ViperGPT](https://viper.cs.columbia.edu/) or [Transformers Agent](https://huggingface.co/docs/transformers/transformers_agents) easier by removing boilerplate prompting code.

### Tools

We can teach language models to call external functions to get additional informations or perform tasks, by encoding the functions' description in the prompt. To avoid duplicating information between the function definition and the description passed to the prompt, we define custom Jinja filters that can extract the function's name, description, signature and source:


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
def agent(tools: List[Callable]):
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

We can instruct models to return their output in a pre-defined format, often JSON. To avoid duplicating information between the function definition and the description passed to the prompt we define a custom Jinja filter that can extract the expected response's schema:

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

## Controlled generation

The first step towards reliability of systems that include large language models is to ensure that there is a well-defined interface between their output and user-defined code. **Outlines** provides ways to control the generation of language models to make their output more predictable.

You can stop the generation after a given sequence has been found:

``` python
answer = model("Tell me a one-sentence joke.", stop_at=["."])
```

You can reduce the completion to a choice between multiple possibilities:

``` python
prompt = labelling("Just awesome", examples)
answer = model(prompt, is_in=["Positive", "Negative"])
```


You can require the generated sequence to be an int or a float:

``` python
import outlines.models as models


model = models.text_completion.hf("sshleifer/tiny-gpt2")
answer = model("2 + 2 = ", type="int")
print(answer)
# 4

model = models.text_completion.hf("sshleifer/tiny-gpt2")
answer = model("1.7 + 3.2 = ", type="float")
print(answer)
# 4.9
```


## Sampling ([uncertainty](https://github.com/normal-computing/outlines/blob/readme/examples/sampling.ipynb), [simulation-based inference](https://github.com/normal-computing/outlines/blob/readme/examples/simulation_based_inference.ipynb))

Outlines is strictly sampling based, and focused on using methods such as [self-consistency](https://arxiv.org/abs/2203.11171), [adaptive consistency](https://arxiv.org/abs/2305.11860), [DiVeRSe](https://arxiv.org/abs/2206.02336), [Tree of thoughts](https://arxiv.org/abs/2305.10601), [lattice sampling](https://arxiv.org/abs/2112.07660), etc. Several samples can be obtained using the `num_samples` keyword argument:

``` python
import outlines.models as models


model = models.text_completion.hf("sshleifer/tiny-gpt2")
answer = model("2 + 2 = ", num_samples=5)
print(answer)
# [4, 5, 4, 4, 4]
```

The focus on sampling allows us to explore different ideas, such as [using the diversity of answers to evaluate the model's uncertainty](https://github.com/normal-computing/outlines/blob/readme/examples/sampling.ipynb), or [simulation-based inference to optimize the prompt](https://github.com/normal-computing/outlines/blob/readme/examples/simulation_based_inference.ipynb).

## Vectorization and parallel execution

You can pass prompts in a NumPy array to Outlines models:

``` python
import numpy as np
import outlines.models as models

model = models.text_completion.openai("text-davinci-003")

prompts = [
    ["Translate 'Hello' in Italian", "Translate 'Hello' in French"],
    ["Translate 'Hello' in Spanish", "Translate 'Hello' in German"],
]
answers = model(prompts)

print(answers.shape)
# (2, 2)
```

Outlines also provide a `outlines.vectorize` decorator that will vectorize any function. If the function is async the requests will be run concurrently:

``` python
import aiohttp
import numpy as np
import outlines

@outlines.vectorize
async def wikipedia_search(query):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={query}&origin=*"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

results = wikipedia_search([["Cat", "Dog"],["Bird", "Horse"]])
print(results.shape)
# (2, 2)
```

This feature allows you to run multiple workflows in parallel, for instance to avoid overfitting when iterating over a workflow or in production to run workflows over several different inputs.

## Contributing

### What contributions?

We curently only accept bug fixes and documentation contributions. If you have a feature request, please start a new [discussions](https://github.com/normal-computing/outlines/discussions). The issue tracker is only intended for actionable items.

### How to contribute?

Run `pip install -e .[test]` or `conda env create -f environment.yml`. To build the documentation you will also need to run `pip install -r requirements-doc.txt`.

Before pushing your code to repository please run `pre-commit run --all-files` and `pytest` to make sure that the code is formatted correctly and that the tests pass.

Do not hesitate to open a draft PR before your contribution is ready, especially if you have questions and/or need feedback.

## Examples

- [Pick the odd one out](https://github.com/normal-computing/outlines/blob/main/examples/pick_odd_one_out.py)
- [Meta prompting](https://github.com/normal-computing/outlines/blob/main/examples/meta_prompting.py)
- [ReAct](https://github.com/normal-computing/outlines/blob/main/examples/meta_prompting.py)
- [Generate code to solve math problems](https://github.com/normal-computing/outlines/blob/main/examples/dust/math-generate-code.py)
- [BabyAGI](https://github.com/normal-computing/outlines/blob/main/examples/babyagi.py)
- [Uncertainty](https://github.com/normal-computing/outlines/blob/main/examples/sampling.ipynb)
- [Simulation-based inference](https://github.com/normal-computing/outlines/blob/main/examples/simulation_based_inference.ipynb)
