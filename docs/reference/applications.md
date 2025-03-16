# Applications

Outlines offers a convenient way of encapsulating a prompt template and an output type in a single object called an `Application`. After instantiating an `Application`, it can be called to generate a response by providing a model and optional arguments used to create the prompt from the template.

## Create an Application

To create an Application, you need to provide 2 arguments:

- A template: either an instance of `outlines.templates.Template` or a callable that takes arguments and returns a prompt
- An output type: a type instance from `outlines.types` that is used to define the structure of the output

```python
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import models, Application
from outlines.types import JsonType
from outlines.templates import Template

class OutputModel(BaseModel):
    result: int

model = models.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

template_string = "What is 2 times {{ num }}?"
template = Template.from_str(template_string)

application = Application(template, JsonType(OutputModel))
```

## Call an Application

Once the Application is instantiated, it can be called. The first argument is the model to use to generate the response. The remaining arguments are passed to the template to create the prompt.

```python
result = application(model, num=3)
print(result) # Expected output: { "result" : 6 }
```
