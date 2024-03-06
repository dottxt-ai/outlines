---
title: Quickstart
---

# Quickstart

After [installing Outlines](installation.md), the fastest way to get to up to speed with the library is to get acquainted with its few core elements. We advise you to take a quick look at this page to see everything Outlines has to offer before diving in the [documentation](reference/index.md).

## Core elements

### Models

The first step when writing a program with Outlines is to initialize a model. Weights will be loaded on the device at this step:

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
```

### Text generator

Once the model is initialized you can build a text generator. This generator can be called with a prompt directly, or you can use the `stream` method to generate text token by token:

=== "Generate"

    ```python
    import outlines

    model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
    generator = outlines.generate.text(model)

    result = generator("What's 2+2?", max_tokens=100)

    print(result)
    # That's right, it's 4! But remember, a delicious and nutrient dense 4,
    # according to YEARS BUILT ON SOLID SCIENCE. This column presents additional
    # findings from the fifteen-year study that produced the 2+2=4 conclusion.
    ```

=== "Stream"

    ```python
    import outlines

    model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
    generator = outlines.generate.text(model)

    stream = generator.stream("What's 2+2?", max_tokens=4)
    for i in range(5):
        token = next(stream)
        print(token)
    # ['Is']
    # [' this']
    # [' even']
    # [' a']
    # [' question']
    ```

### Multi-label classification

Outlines allows you to do multi-label classification by guiding the model so it can only output either of the specified choices:

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.choice(model, ["Blue", "Red", "Yellow"])

color = generator("What is the closest color to Indigo? ")
print(color)
# Blue
```

### JSON-structured generation

Outlines can guide models so that they output valid JSON **100%** of the time. You can either specify the structure using [Pydantic][pydantic]{:target="_blank"} or a string that contains a [JSON Schema][jsonschema]{:target="_blank"}:

=== "Pydantic"

    ```python
    from enum import Enum
    from pydantic import BaseModel, constr, conint

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

    model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
    generator = outlines.generate.json(model, Character)

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

    model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
    generator = outlines.generate.json(model, schema)
    character = generator(
        "Generate a new character for my awesome game: "
        + "name, age (between 1 and 99), armor and strength. "
        )
    print(character)
    # {'name': 'Yuki', 'age': 24, 'armor': 'plate', 'strength': 3}
    ```

!!! Note

    We advise you to constrain the length of the strings fields when first testing your schema, especially with small models.

### Grammar-structured generation

Outlines also allows to generate text that is valid to any [context-free grammar][cfg]{:target="_blank"} (CFG) in the [EBNF format][ebnf]{:target="_blank"}. Grammars can be intimidating, but they are a very powerful tool! Indeed, they determine the syntax of every programming language, valid chess moves, molecule structure, can help with procedural graphics generation, etc.

Here we show a simple example of a grammar that defines arithmetic operations:

```python
from outlines import models, generate

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

model = models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = generate.cfg(model, arithmetic_grammar, max_tokens=100)

result = generator("Question: How can you write 5*5 using addition?\nAnswer:")
print(result)
# 5+5+5+5+5
```


EBNF grammars can be cumbersome to write. This is why Outlines provides grammar definitions in the `outlines.grammars.` module

```python
from outlines import models, generate, grammars

model = models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = generate.cfg(model, grammars.arithmetic, max_tokens=100)

result = generator("Question: How can you write 5*5 using addition?\nAnswer:")
print(result)
# 5+5+5+5+5
```

The available grammars are listed [here](https://github.com/outlines-dev/outlines/tree/main/outlines/grammars).


### Regex-structured generation

Slightly simpler, but no less useful, Outlines can generate text that is in the language of a [regular expression](https://www.regular-expressions.info/tutorial.html). For instance to force the model to generate IP addresses:

```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

regex_str = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
generator = generate.regex(model, regex_str)

result = generator("What is the IP address of localhost?\nIP: ")
print(result)
# 127.0.0.100
```

### Generate a given Python type

We provide a shortcut to regex-structured generation for simple use cases. Pass a Python type to the `outlines.generate.format` function and the LLM will output text that matches this type:

```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = generate.format(model, int)

result = generator("What is 2+2?")
print(result)
# 4
```

## Deploy using vLLM and FastAPI

Outlines can be deployed as a LLM service using [vLLM][vllm]{:target="_blank"} and [FastAPI][fastapi]{:target="_blank"}. The server supports asynchronous processing of incoming requests, and benefits from the performance of vLLM.

First start the server:

```python
python -m outlines.serve.serve --model="mistralai/Mistral-7B-Instruct-v0.2"
```

Or you can start the server with Outlines' official Docker image:

```bash
docker run -p 8000:8000 outlinesdev/outlines --model="mistralai/Mistral-7B-Instruct-v0.2"
```

This will by default start a server at `http://127.0.0.1:8000` (check what the console says, though). Without the `--model` argument set, the OPT-125M model is used.


You can then query the model in shell by passing a prompt and a [JSON Schema][jsonschema]{:target="_blank"} specification for the structure of the output:

```bash
curl http://127.0.0.1:8000/generate \
    -d '{
        "prompt": "Question: What is a language model? Answer:",
        "schema": {"type": "string"}
        }'
```

Or use the [requests][requests]{:target="_blank"} library from another python program. You can read the [vLLM documentation][vllm]{:target="_blank"} for more details.

## Utilities

### Prompt templates

Prompting can lead to messy code. Outlines' prompt functions are python functions that contain a template for the prompt in their docstring. We use a powerful templating language to allow you to loop over lists, dictionaries, add conditionals, etc. directly from the prompt. When called, a prompt function returns the rendered template:

```python
import outlines

@outlines.prompt
def few_shots(instructions, examples, question):
    """{{ instructions }}

    Examples
    --------

    {% for example in examples %}
    Q: {{ example.question }}
    A: {{ example.answer }}

    {% endfor %}
    Question
    --------

    Q: {{ question }}
    A:
    """

instructions = "Please answer the following question following the examples"
examples = [
    {"question": "2+2=?", "answer":4},
    {"question": "3+3=?", "answer":6}
]
question = "4+4 = ?"

prompt = few_shots(instructions, examples, question)
print(prompt)
# Please answer the following question following the examples

# Examples
# --------

# Q: 2+2=?
# A: 4

# Q: 3+3=?
# A: 6

# Question
# --------

# Q: 4+4 = ?
# A:
```

### Outlines functions

Once you are done experimenting with a prompt and an output structure, it is useful to be able to encapsulate all of these in a single function that can be called from other parts of the program. This is what `outlines.Function` allows you to do:

=== "function.py"

    ```python
    from pydantic import BaseModel

    import outlines


    @outlines.prompt
    def tell_a_joke(topic):
        """Tell me a joke about {{ topic }}."""

    class Joke(BaseModel):
        setup: str
        punchline: str

    generate_joke = outlines.Function(
        tell_a_joke,
        Joke,
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    ```

=== "Call a function"

    ```python
    from .function import generate_joke

    response = generate_joke("baseball")

    # haha
    # Joke(setup='Why was the baseball in a bad mood?', punchline='Because it got hit around a lot.')
    ```

=== "Call a function stored on GitHub"

    You can load a function that is stored on a repository on GitHub directly from Outlines. Say `Someone` stores a function in `joke.py` at the root of the `TheirRepo` repository:

    ```python
    import outlines

    joke = outlines.Function.from_github("Someone/TheirRepo/joke")
    response = joke("baseball")
    ```
    It make it easier for the community to collaborate on the infinite number of use cases enabled by these models!

## Going further

If you need more inspiration you can take a look at the [cookbook](cookbook/index.md). If you have any question, or requests for documentation please reach out to us on [GitHub](https://github.com/outlines-dev/outlines/discussions), [Twitter](https://twitter.com/remilouf) or [Discord](https://discord.gg/UppQmhEpe8).


[pydantic]: https://docs.pydantic.dev/latest
[jsonschema]: https://json-schema.org/
[fastapi]: https://fastapi.tiangolo.com/
[cfg]: https://en.wikipedia.org/wiki/Context-free_grammar
[ebnf]: https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
[requests]: https://requests.readthedocs.io/en/latest/
[vllm]: https://docs.vllm.ai/en/latest/index.html
