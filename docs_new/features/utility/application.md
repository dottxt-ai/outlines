---
title: Application
---

# Application

The `Application` class enables you to encapsulate a prompt template and an output type into a reusable component.

## Overview

To create an `Application` instance, initiliaze the class with a prompt template and an output type. You can then call the application with a model and the variables defined in your template as keyword arguments.

For instance:

```python
from typing import Literal
from outlines import Application, Template, from_transformers

# Create a template
template_str = "Is {{ name }} a boy or a girl name?"""
template = Template.from_string(template_str)

# Create a model
model = from_transformers(...)

# Create the application and call it to generate text
application = Application(template, Literal["boy", "girl"])
reponse = application(model, name="Alice")
print(response) # "girl"
```

Instead of providing an Outlines `Template` instance, you can provide a `Callable` that returns a string. The parameters of the callable are used as the variables of the template such that you must provide values for them when calling the application.

For instance, the following function would be equivalent to the template defined above when used in an application:

```python
def template(name: str) -> str:
    return f"Is {name} a boy or a girl name?"
```
