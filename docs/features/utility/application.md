---
title: Application
---

# Application

The `Application` class enables you to encapsulate a prompt template and an output type into a reusable component.

## Overview

An `Application` combines a prompt template with an output type, creating a reusable component that can be applied to different models.

Applications are useful for simplifying repeated tasks where you have a well-defined `Template` and a fixed output type, such as classification tasks or data extraction.

To create an `Application` instance, initialize the class with a prompt template and an output type. You can then call the application with a model and the variables defined in your template in a dictionary.

For instance:

```python
from typing import Literal
import transformers
from outlines import Application, Template, from_transformers

# Create a template
template_str = "Is {{ name }} a boy or a girl name?"""
template = Template.from_string(template_str)

# Create a model
model = from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    transformers.AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

# Create the application and call it to generate text
application = Application(template, Literal["boy", "girl"])
response = application(model, {"name": "Alice"}, max_new_tokens=10)

print(response) # "girl"
```

Instead of providing an Outlines `Template` instance, you can provide a `Callable` that returns a string. The parameters of the callable are used as the variables of the template such that you must provide values for them in the dictionary when calling the application.

For instance, we can create the same example as above using a a function instead of a template:

```python
from typing import Literal
import transformers
from outlines import Application, from_transformers

# Create a function that will be used as a template
def template(name: str) -> str:
    return f"Is {name} a boy or a girl name?"

# Create a model
model = from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    transformers.AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

# Create the application with the function template and call it to generate text
application = Application(template, Literal["boy", "girl"])
response = application(model, {"name": "Alice"}, max_new_tokens=10)

print(response) # "girl"
```
