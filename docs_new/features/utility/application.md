---
title: Application
---

# Application

The `Application` class enables you to encapsulate a prompt template and an output type into a reusable component.

## Overview

To create an `Application` instance, initiliaze the class with a prompt template and an output type. You can then call the application with a model and the variables defined in your template in a dictionary.

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

For instance, the following function would be equivalent to the template defined above when used in an application:

```python
def template(name: str) -> str:
    return f"Is {name} a boy or a girl name?"
```
