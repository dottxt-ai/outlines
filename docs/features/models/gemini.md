# Gemini

!!! Installation

    You need to install the `google.genai` libray to be able to use the Gemini API in Outlines. Install all optional dependencies of the `Gemini` model with: `pip install outlines[gemini]`.

    You also need to have a Gemini API key. This API key must either be set as an environment variable called `GEMINI_API_KEY` or be provided to the `google.genai.Client` class when instantiating it.

## Model Initialization

To create a Gemini model instance, you can use the `from_gemini` function. It takes 2 arguments:

- `client`: a `google.genai.Client` instance
- `model_name`: the name of the model you want to use in subsequent model calls (optional)

For instance:

```python
import outlines
from google import genai

# Create the client
client = genai.Client()

# Create the model
model = outlines.from_gemini(
    client,
    "gemini-1.5-flash-latest"
)
```

Check the [Gemini documentation](https://github.com/googleapis/python-genai) for an up-to-date list of available models.

## Text Generation

Once you've created your Outlines `Gemini` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt.

For instance:

```python
import outlines
from google import genai

# Create the model
model = outlines.from_gemini(genai.Client(), "gemini-1.5-flash-latest")

# Call it to generate text
result = model("What's the capital of Latvia?", max_output_tokens=20)
print(result) # 'Riga'
```

Gemini also supports streaming. For instance:

```python
import outlines
from google import genai

# Create the model
model = outlines.from_gemini(genai.Client(), "gemini-1.5-flash-latest")

# Stream text
for chunk in model.stream("Write a short story about a cat.", max_output_tokens=20):
    print(chunk) # 'In...'
```

## Structured Generation

Gemini provides supports for some forms of structured output: multiple choice, JSON schema (with caveats) and lists of structured objects. To use it, call the model with an `output_type` on top of your prompt.

#### Multiple Choice

```python
import outlines
from google import genai
from enum import Enum

class PizzaOrBurger(Enum):
    pizza = "pizza"
    burger = "burger"

# Create the model
model = outlines.from_gemini(genai.Client(), "gemini-1.5-flash-latest")

# Call it with the ouput type to generate structured text
result = model("Pizza or burger?", PizzaOrBurger, max_output_tokens=20)
print(result) # 'pizza'
```

#### JSON Schema

Gemini supports only three types of objects used to define a JSON Schema:

- Pydantic classes
- Dataclasses
- TypedDicts

```python
from typing import List
from pydantic import BaseModel
from google import genai
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_gemini(genai.Client(), "gemini-1.5-flash-latest")

# Call it with the ouput type to generate structured text
result = model("Create a character", Character)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

#### Lists of Structured Objects

A specificity of Gemini is that, despite not supporting regex, it does support a list of structured objects as an output type. To use it, put any of three available types described above in the typing `List` class

```python
from dataclasses import dataclass
from google import genai
import outlines

@dataclass
class Character:
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_gemini(genai.Client(), "gemini-1.5-flash-latest")

# Call it with the ouput type to generate structured text
result = model("Create a character", list[Character])
print(result) # '[{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}, {["name":...'
```

!!! Attention

    The structured objects must be in a built-in `list`, not a `List` from the `typing` library

## Inference arguments

You can provide the same optional parameters you would pass to the `google.genai.Client` client both during the initialization of the Gemini model and when generating text. Some of the common inference arguments include `max_output_tokens`, `temperature`, and other generation parameters.

Consult the [Google Generative AI documentation](https://github.com/googleapis/python-genai) for the full list of parameters.
