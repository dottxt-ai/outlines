---
title: Template
---

# Template

Outlines templates provide a way of creating reusable prompt structures with placeholders for dynamic content.

## Overview

To create a `Template` instance, you can use two class methods:
- `from_string`: Creates a template from a string containing a Jinja2 template
- `from_file`: Creates a template from a file containing a Jinja2 template

After creating a template, you can call it with the variables required by the template as keyword arguments.

For instance:

```python
from outlines import Template

# Create a template from a string
template_str = """
Hello, {{ name }}!
The weather today is {{ weather }}.
"""
template = Template.from_string(template_str)

# Create a template from a file, assuming the content of template_str is put into a file
template = Template.from_file("path_to/my_file.txt")

# Call the template to render the prompt
prompt: str = template(name="Alice", weather="sunny")
print(prompt)  # "Hello, Alice!\nThe weather today is sunny."
```

## Composite Templates

Templates can be nested and composed to create complex prompt structures:

```python
from outlines import Template

# Create component templates
user_template = Template.from_string("User: {{ query }}")
system_template = Template.from_string("System: {{ instruction }}")

# Create a composite template
chat_template = Template.from_string("""
{{ system }}
{{ user }}
""")

# Fill in nested templates
prompt = chat_template(
    system=system_template(instruction="You are a helpful assistant."),
    user=user_template(query="What is machine learning?")
)

print(prompt)
# System: You are a helpful assistant.
#
# User: What is machine learning?
```

## Custom Filters

You can add custom filters to your Outlines template to extend the templating functionality. To do so, provide as second argument a dictionary with filter names as keys and filter functions as values. The filter can then be used in your jinja2 template following the regular syntax. When rendering a prompt, the function will be applied to the associated variable.

For instance:

```python
from outlines import Template

def uppercase(text: str) -> str:
    return text.upper()

# Add custom filter when creating template
template = Template.from_string(
    "Hello {{ name | uppercase }}!",
    filters={"uppercase": uppercase}
)
prompt = template(name="alice")
print(prompt)  # "Hello ALICE!"
```
