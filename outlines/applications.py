"""Encapsulate a prompt template and an output type into a reusable object."""

from typing import Any, Callable, Dict, Optional, Union

from outlines.generator import (
    BlackBoxGenerator,
    Generator,
    SteerableGenerator,
)
from outlines.models.base import Model
from outlines.templates import Template


class Application:
    """
    Application is a class that encapsulates a prompt template and an
    output type. It can be called to generate a response by providing a
    model, the values to be substituted in the template in a dictionary
    and optional inference parameters.

    Parameters
    ----------
    template : Union[Template, Callable]
        A callable that takes arguments and returns a prompt string.
    output_type : Any
        The expected output type of the generated response.

    Examples
    --------
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
    template = Template.from_string(template_string)

    application = Application(template, JsonType(OutputModel))

    result = application(model, {"num": 3}, max_new_tokens=20)
    print(result)  # Expected output: { "result" : 6 }
    ```

    """
    def __init__(
        self,
        template: Union[Template, Callable],
        output_type: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        template
            The template to use to build the prompt.
        output_type
            The output type provided to the generator.

        """
        self.template = template
        self.output_type = output_type
        self.generator: Optional[Union[
            BlackBoxGenerator, SteerableGenerator
        ]] = None
        self.model: Optional[Model] = None

    def __call__(
        self,
        model: Model,
        template_vars: Dict[str, Any],
        **inference_kwargs
    ) -> Any:
        """
        Parameters
        ----------
        model
            The model to use to generate the response.
        template_vars
            The variables to be substituted in the template.
        **inference_kwargs
            Additional keyword arguments to pass to the model.
        Returns
        -------
        Any
            The generated response.
        """
        if model is None:
            raise ValueError("you must provide a model")
        # We save the generator to avoid creating a new one for each call.
        # If the model has changed since the last call, we create a new
        # generator.
        if model != self.model:
            self.model = model
            self.generator = Generator(model, self.output_type)  # type: ignore

        prompt = self.template(**template_vars)
        assert self.generator is not None
        return self.generator(prompt, **inference_kwargs)
