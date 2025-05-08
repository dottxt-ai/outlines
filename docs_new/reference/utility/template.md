---
title: Template API
---

# Template

::: outlines.templates.Template

Templates provide a way to create reusable prompt structures with placeholders for dynamic content.

## Basic Usage

```python
from outlines import Template

# Create a simple template
template = Template("Hello, {name}! The weather today is {weather}.")

# Fill in the template
prompt = template.format(name="Alice", weather="sunny")
print(prompt)  # Output: "Hello, Alice! The weather today is sunny."
```

## Composing Templates

Templates can be nested and composed:

```python
from outlines import Template

# Create component templates
user_template = Template("User: {query}")
system_template = Template("System: {instruction}")

# Create a composed template
chat_template = Template("""
{system}

{user}
""")

# Fill in nested templates
prompt = chat_template.format(
    system=system_template.format(instruction="You are a helpful assistant."),
    user=user_template.format(query="What is machine learning?")
)

print(prompt)
# Output:
# System: You are a helpful assistant.
#
# User: What is machine learning?
```

## Using with Generator

Templates are useful when combined with the Generator class:

```python
from outlines import Template, Generator, from_transformers

# Create a template
template = Template("""
Translate the following {source_language} text to {target_language}:

{text}

Translation:
""")

# Initialize a model
model = from_transformers(...)

# Create a generator
generator = Generator(model)

# Use the template with the generator
translation = generator(
    template.format(
        source_language="English",
        target_language="French",
        text="Hello, how are you today?"
    )
)

print(translation)  # Output: "Bonjour, comment allez-vous aujourd'hui?"
```

## Multimodal Templates

For multimodal inputs, use the `Vision` class:

```python
from outlines import Template, Generator, from_transformers, Vision
from PIL import Image

# Create a vision template
template = Template("""
Describe what you see in this image: {image}
""")

# Load an image
image = Image.open("example.jpg")

# Format the template with the image
prompt = template.format(image=Vision(image))

# Initialize a multimodal model
model = from_transformers(...)

# Generate a description
generator = Generator(model)
description = generator(prompt)

print(description)
```
