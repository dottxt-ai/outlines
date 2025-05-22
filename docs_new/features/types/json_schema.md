---
title: JSON Schema
---

# JSON Schema

::: outlines.types.json_schema

JSON Schema provides a way to define the structure of JSON data that you want to generate. Outlines supports multiple ways to define JSON schemas.

## Basic Usage

```python
from outlines import Generator, from_transformers
from outlines.types import JsonSchema

# Define a simple JSON schema
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "age"]
}

# Create a JSON schema output type
json_output = JsonSchema(person_schema)

# Initialize a model
model = from_transformers(...)

# Create a generator with the JSON schema output type
generator = Generator(model, json_output)

# Generate JSON data
prompt = "Generate information for a fictional person:"
result = generator(prompt)

print(result)
# Example output: {"name": "John Doe", "age": 32, "email": "john.doe@example.com"}
```

## Using Pydantic Models

```python
from pydantic import BaseModel
from typing import List, Optional
from outlines import Generator, from_transformers

# Define a Pydantic model
class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

# Initialize a model
model = from_transformers(...)

# Create a generator with the Pydantic model
generator = Generator(model, Person)

# Generate JSON data
prompt = "Generate information for a fictional person:"
result = generator(prompt)

# Parse the result into a Pydantic model
person = Person.model_validate_json(result)
print(f"Name: {person.name}, Age: {person.age}")
```

## Other Schema Types

Outlines also supports:

- TypedDict
- Dataclasses
- Direct JSON Schema objects

For more detailed examples, see the [JSON Generation guide](/user_guide/structured_generation/json.md).
