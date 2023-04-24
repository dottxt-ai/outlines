import functools
from dataclasses import dataclass
from typing import Callable, Optional, Union

from pydantic import BaseModel

FunctionType = type(lambda x: None)
BaseModelType = type(BaseModel)


@dataclass
class function:
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
    return validator.parse_raw(result)


@validate.register(FunctionType)
def validate_function(validator, result):
    return validator(result)
