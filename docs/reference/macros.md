# Macros

Outlines offers a convenient way of encapsulating a model, a prompt template, and an output type in a single object called a `Macro`. After instantiating a `Macro`, it can be called just like a function with arguments that will be passed to the template to create the prompt. The prompt is then used to call the model with the output type first specified to generate an answer.

## Create a Macro

To create a Macro, you need to provide 3 arguments:

- A model: an instance of an Outlines model class from module `outlines.models`
- A template: either an instance of `outlines.templates.Template` or a callable that takes arguments and returns a prompt
- An output type: a type instance from `outlines.types` that is used to define the structure of the output

```python
from pydantic import BaseModel
from outlines.models import transformers
from outlines.templates import Template
from outlines.types import JsonType

class OutputModel(BaseModel):
    result: int

model = transformers.from_transformers(
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-4k-instruct"
)
template = Template.from_str("What is 2 times {{ num }}?")
output_type = JsonType(OutputModel)

macro = Macro(model, template, output_type)
```

## Call a Macro

Once the Macro is instantiated, it can be called just like a function with arguments that will be passed to the template to create the prompt. The prompt is then used to call the model with the output type first specified to generate an answer.

```python
result = macro(num=3)
print(result) # Expected output: { "result" : 6 }
```
