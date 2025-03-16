from typing import Any, Callable, Optional, Union

from outlines.generator import BlackBoxGenerator, Generator, SteerableGenerator
from outlines.templates import Template
from outlines.models import BlackBoxModel, SteerableModel


class Application:
    """
    Application is a class that encapsulates a prompt template and an
    output type. It can be called to generate a response by providing a
    model and optional arguments used to create the prompt from the template.

    Parameters
    ----------
    template : Union[Template, Callable]
        A callable that takes arguments and returns a prompt string.
    output_type : Any
        The expected output type of the generated response.

    Examples
    --------
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

    result = application(model, num=3)
    print(result)  # Expected output: { "result" : 6 }
    """
    def __init__(self, template: Union[Template, Callable], output_type: Any):
        self.template = template
        self.output_type = output_type
        self.model: Optional[Union[BlackBoxModel, SteerableModel]] = None
        self.generator: Optional[Union[BlackBoxGenerator, SteerableGenerator]] = None

    def __call__(self, model: Union[BlackBoxModel, SteerableModel], *args, **kwargs):
        if model is None:
            raise ValueError("you must provide a model")
        # We save the generator to avoid creating a new one for each call.
        # If the model has changed since the last call, we create a new
        # generator.
        elif model != self.model:
            self.model = model
            self.generator = Generator(model, self.output_type)

        prompt = self.template(*args, **kwargs)
        assert self.generator is not None
        return self.generator(prompt)
