import functools
from dataclasses import dataclass
from typing import Callable, Optional, Union

from pydantic import BaseModel

FunctionType = type(lambda x: None)
BaseModelType = type(BaseModel)


@dataclass
class function:
    """Represents a function that uses a language model to generate its output.

    When called, the `function` instance passes the arguments to the prompt
    function, the rendered prompt is passed to the language model, and its
    result to an (optional) validation function.

    Attributes
    ----------
    model
        A function that takes a string and returns a string that contains the
        model's return value.
    prompt
        A prompt-generating function.
    validator
        A function that takes the output of the language model, parses it and
        returns it in a normalized format.

    """

    model: Callable
    prompt: Callable
    validator: Optional[Union[Callable, BaseModel]] = None

    def __call__(self, *args, **kwargs):
        rendered_prompt = self.prompt(*args, **kwargs)
        result = self.model(rendered_prompt)
        validated_result = validate(self.validator, result)
        return validated_result


@functools.singledispatch
def validate(validator, result):
    if validator is not None:
        raise NotImplementedError(
            f"Cannot validate the input with validator of type {type(validator)}"
        )
    else:
        return result


@validate.register(BaseModelType)
def validate_pydantic(validator, result):
    if hasattr(validator, "model_validate_json"):
        return validator.model_validate_json(result)
    else:  # pragma: no cover
        return validator.parse_raw(result)


@validate.register(FunctionType)
def validate_function(validator, result):
    return validator(result)
