# Make the LLM follow a JSON Schema

Outlines can make any open source model return a JSON object that follows a structure that is specified by the user. This is useful whenever we want the output of the model to be processed by code downstream: code does not understand natural language but rather the structured language it has been programmed to understand.

There are mostly two reasons why someone would want to get an output formatted as JSON from a LLM:

1. Parse the answer (e.g. with Pydantic), store it somewhere, return it to a user, etc.
2. Call a function with the result

Outlines has you covered in both cases! Indeed, to define the structure of the JSON you want the model to follow you can either provide a Pydantic model, or a function. No need to duplicate code!

## Using Pydantic

Outlines can infer the structure of the output from a Pydantic model. The result is an instance of the model that contains the values returned by the LLM:

```python
from pydantic import BaseModel

from outlines import models
from outlines import text


class User(BaseModel):
    name: str
    last_name: str
    id: int


model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = text.generate.json(model, User)
result = generator("Create a user profile with the fields name, last_name and id")
print(result)
# User(name="John", last_name="Doe", id=11)
```

!!! warning "JSON and whitespaces"

    By default Outlines lets model choose the number of linebreaks and white spaces used to structure the JSON. Small models tend to struggle with this, in which case we recommend to set the value of the parameter `whitespace_pattern` to the empty string:

    ```python
    generator = text.generate.json(model, User, whitespace_pattern="")
    ```

## From a function's signature

Outlines can infer the structure of the output from the signature of a function. The result is a dictionary, and can be passed directly to the function using the usual dictionary expansion syntax `**`:

```python
from outlines import models
from outlines import text

def add(a: int, b: int):
    return a + b

model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = text.generate.json(model, add)
result = generator("Return two integers named a and b respectively. a is odd and b even.")

print(add(**result))
# 3
```

A great advantage of passing functions directly to specify the structure is that the structure of the LLM will change with the function's definition. No need to change the code at several places!
