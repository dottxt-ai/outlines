from dataclasses import dataclass
from typing import Any, Callable, Union

from outlines.generate import Generator
from outlines.templates import Template
from outlines.models import Model


@dataclass
class Macro:
    """
    Macro is a class that encapsulates a model, a prompt template, and an
    output type. It can be called to generate a response.

    Parameters
    ----------
    model : Model
        The Outlines model to be used for generating responses.
    template : Union[Template, Callable]
        A callable that takes arguments and returns a prompt string.
    output_type : Any
        The expected output type of the generated response.

    Examples
    --------
    from pydantic import BaseModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from outlines import models, Macro
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

    my_macro = Macro(model, template, JsonType(OutputModel))

    result = my_macro(num=3)
    print(result)  # Expected output: { "result" : 6 }
    """
    model: Model
    template: Union[Template, Callable]
    output_type: Any

    def __post_init__(self):
        self.template = self.template
        self.generator = Generator(self.model, self.output_type)

    def __call__(self, *args, **kwargs):
        prompt = self.template(*args, **kwargs)
        return self.generator(prompt)
