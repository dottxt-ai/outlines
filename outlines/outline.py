import json
from dataclasses import dataclass

import jsonschema
from pydantic import BaseModel

from outlines import generate


@dataclass
class Outline:
    """
    Outline is a class that creates a callable object to generate responses
    based on a given model, a prompt template (a function that returns a `str`) and an expected output type.

    Parameters
    ----------
    model : object
        The model to be used for generating responses.
    template : function
        A function that takes arguments and returns a prompt string.
    output_type : type
        The expected output type of the generated response.

    Examples
    --------
    from pydantic import BaseModel
    from outlines import models, Outline

    class OutputModel(BaseModel):
        result: int

    model = models.transformers("gpt2")

    def template(a: int) -> str:
        return f"What is 2 times {a}?"

    fn = Outline(model, template, OutputModel)

    result = fn(3)
    print(result)  # Expected output: OutputModel(result=6)
    """

    def __init__(self, model, template, output_type):
        if isinstance(output_type, str):
            try:
                jsonschema.Draft7Validator.check_schema(json.loads(output_type))
            except jsonschema.exceptions.SchemaError as e:
                raise TypeError(f"Invalid JSON Schema: {e.message}")
        elif not issubclass(output_type, BaseModel):
            raise TypeError(
                "output_type must be a Pydantic model or a valid JSON Schema string"
            )

        self.template = template
        self.output_type = output_type
        self.generator = generate.json(model, output_type)

    def __call__(self, *args, **kwargs):
        prompt = self.template(*args, **kwargs)
        response = self.generator(prompt)
        try:
            if isinstance(self.output_type, str):
                return json.loads(response)
            return self.output_type.parse_raw(response)
        except (ValueError, SyntaxError):
            # If `outlines.generate.json` works as intended, this error should never be raised.
            raise ValueError(f"Unable to parse response: {response.strip()}")
