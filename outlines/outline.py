import ast
from dataclasses import dataclass


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
    from outlines import models

    model = models.transformers("gpt2")

        def template(a: int) -> str:
            return f"What is 2 times {a}?"

        fn = Outline(model, template, int)

        result = fn(3)
        print(result)  # Expected output: 6
    """

    def __init__(self, model, template, output_type):
        self.model = model
        self.template = template
        self.output_type = output_type

    def __call__(self, *args):
        prompt = self.template(*args)
        response = self.model.generate(prompt)
        try:
            parsed_response = ast.literal_eval(response.strip())
            if isinstance(parsed_response, self.output_type):
                return parsed_response
            else:
                raise ValueError(
                    f"Response type {type(parsed_response)} does not match expected type {self.output_type}"
                )
        except (ValueError, SyntaxError):
            raise ValueError(f"Unable to parse response: {response.strip()}")
