# Gemini

!!! Installation

    You need to install the `google-generativeai` library to be able to use the Gemini API in Outlines: `pip install google-generativeai`

    You also need to have a Gemini API key. This API key must either be set as an environment variable called `GEMINI_API_KEY` or be provided to the `google.generativeai.GenerativeModel` class when instantiating it.

## Model Initialization

To create a Gemini model instance, you can use the `from_gemini` function. It takes 2 arguments:

- `client`: a `google.generativeai.GenerativeModel` instance
- `model_name`: the name of the model you want to use in subsequent model calls (optional)

For instance:

```python
import outlines
import google.generativeai as genai

# Create the client
client = genai.GenerativeModel()

# Create the model
model = outlines.from_gemini(
    client,
    "gemini-1-5-flash"
)
```

Check the [Gemini documentation](https://github.com/googleapis/python-genai) for an up-to-date list of available models.

## Text Generation

Once you've created your Outlines `Gemini` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt.

For instance:

```python
import outlines
import google.generativeai as genai

# Create the model
model = outlines.from_gemini(genai.GenerativeModel(), "gemini-1-5-flash")

# Call it to generate text
result = model("What's the capital of Latvia?", max_tokens=20)
print(result) # 'Riga'
```

Gemini also supports streaming. For instance:

```python
import outlines
import google.generativeai as genai

# Create the model
model = outlines.from_gemini(genai.GenerativeModel(), "gemini-1-5-flash")

# Stream text
for chunk in model.stream("Write a short story about a cat.", max_tokens=20):
    print(chunk) # 'In...'
```

## Structured Generation

Gemini provides supports for some forms of structured output: multiple choice, JSON schema (with caveats) and lists of structured objects. To use it, call the model with an `output_type` on top of your prompt.

#### Multiple Choice

```python
import outlines
import google.generativeai as genai
from enum import Enum

class PizzaOrBurger(Enum):
    pizza = "pizza"
    burger = "burger"

# Create the model
model = outlines.from_gemini(genai.GenerativeModel(), "gemini-1-5-flash")

# Call it with the ouput type to generate structured text
result = model("Pizza or burger?", PizzaOrBurger, max_tokens=20)
print(result) # 'pizza'
```

#### JSON Schema

Gemini supports only three types of objects used to define a JSON Schema:

- Pydantic classes
- Dataclasses
- TypedDicts

```python
import google.generativeai as genai
import outlines
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_gemini(genai.GenerativeModel(), "gemini-1-5-flash")

# Call it with the ouput type to generate structured text
result = model("Create a character", Character)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

#### Lists of Structured Objects

A specificity of Gemini is that, despite not supporting regex, it does support a list of structured objects as an output type. To use it, put any of three available types described above in the typing `List` class

```python
from dataclasses import dataclass
from typing import List
import google.generativeai as genai
import outlines

@dataclass
class Character:
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_gemini(genai.GenerativeModel(), "gemini-1-5-flash")

# Call it with the ouput type to generate structured text
result = model("Create a character", List[Character])
print(result) # '[{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}, {["name":...'
```

## Inference arguments

You can provide the same optional parameters you would pass to the `google.generativeai.GenerativeModel` client both during the initialization of the Gemini model and when generating text. Some of the common inference arguments include `max_tokens`, `temperature`, and other generation parameters.

Consult the [Google Generative AI documentation](https://github.com/googleapis/python-genai) for the full list of parameters.
