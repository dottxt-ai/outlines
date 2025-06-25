### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import json as pyjson
import warnings
from typing import Callable, Optional, Union

from pydantic import BaseModel

from outlines.models.openai import OpenAI
from outlines.types.dsl import JsonSchema
from outlines.types.utils import get_schema_from_signature, is_callable
from outlines.v0_legacy.samplers import Sampler, multinomial
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.models.transformers_vision import TransformersVision


def json(
    model,
    schema_object: Union[str, object, Callable],
    sampler: Sampler = multinomial(),
    whitespace_pattern: Optional[str] = None,
) -> GeneratorV0Adapter:
    """Generate structured text that follows a JSON Schema.

    This function is deprecated starting from v1.0.0. Do not use it.
    Support for it will be removed in v1.1.0.
    Use the `Generator` object instead:

    ```python
    from outlines import Generator
    from outlines.types import JsonSchema
    generator = Generator(model, JsonSchema(schema_object))
    ```

    You can then call the generator created with a prompt to generate text that
    matches the JSON Schema.

    Parameters
    ----------
    model:
        The Outlines v0 model to use to generate text.
    schema_object:
        The JSON Schema to generate data for. Can be a JSON string, a Pydantic
        model, or a callable that returns a JSON schema.
    sampler:
        The sampler defining the sampling parameters.
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string
        literals).
        Example: allow only a single space or newline with
        `whitespace_pattern=r"[\n ]?"`.

    Returns
    -------
    A `GeneratorV0Adapter` instance that can be called with a prompt to
    generate text.

    """
    warnings.warn("""
        The `json` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Use the `Generator` object instead:
        ```python
        from outlines import Generator
        from outlines.types import JsonSchema
        schema_str = '...'  # JSON Schema as a string
        generator = Generator(model, JsonSchema(schema_str))
        ```
        You can then call the generator created with a prompt to generate
        JSON data that matches the schema.
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, OpenAI):
        return json_openai(model, schema_object, sampler)

    if is_callable(schema_object):
        json_schema = JsonSchema(get_schema_from_signature(schema_object))  # type: ignore
    else:
        json_schema = JsonSchema(schema_object, whitespace_pattern)

    if isinstance(model, TransformersVision):
        generator = GeneratorVisionV0Adapter(model, json_schema, sampler)
    else:
        generator = GeneratorV0Adapter(model, json_schema, sampler)  # type: ignore

    if isinstance(schema_object, type(BaseModel)):
        setattr(
            generator,
            'format_sequence',
            lambda x: schema_object.model_validate_json(x)
        )
    else:
        setattr(generator, 'format_sequence', lambda x: pyjson.loads(x))

    return generator


def json_openai(
    model,
    schema_object: Union[str, object, Callable],
    sampler: Sampler = multinomial(),
):
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            "The OpenAI API does not support any other sampling algorithm "
            + "than the multinomial sampler."
        )

    if isinstance(schema_object, type(BaseModel)):
        schema = schema_object.model_json_schema()
        schema["additionalProperties"] = False
        schema = pyjson.dumps(schema)
        format_sequence = lambda x: schema_object.model_validate_json(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object or a string that contains the JSON Schema "
            + "specification"
        )

    # create copied, patched model with normalized json schema set
    generator = model.new_with_replacements(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "default",
                "strict": True,
                "schema": pyjson.loads(schema),
            },
        }
    )

    generator.format_sequence = format_sequence

    return generator
