### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import re
import json as pyjson
import warnings
from enum import Enum
from typing import Callable, List, Union

from outlines.models.openai import OpenAI
from outlines.types.utils import get_schema_from_enum
from outlines.v0_legacy.generate.api import GeneratorV0Adapter, GeneratorVisionV0Adapter
from outlines.v0_legacy.generate.json import json
from outlines.v0_legacy.generate.regex import regex
from outlines.v0_legacy.samplers import Sampler, multinomial
from outlines_core.fsm.json_schema import build_regex_from_schema


def choice(
    model,
    choices: Union[List[str], type[Enum]],
    sampler: Sampler = multinomial(),
) -> Union[GeneratorV0Adapter, GeneratorVisionV0Adapter, Callable]:
    """Generate a choice from a list of options.

    This function is deprecated starting from v1.0.0. Do not use it.
    Support for it will be removed in v1.1.0.
    Use the `Generator` object instead:

    ```python
    from typing import Literal
    from outlines import Generator
    generator = Generator(model, Literal["foo", "bar"])
    ```

    You can then call the generator created with a prompt to generate a
    choice from the list of options.

    Parameters
    ----------
    model:
        The Outlines v0 model to use to generate text.
    choices:
        A list of options to choose from.
    sampler:
        The sampler defining the sampling parameters.

    Returns
    -------
    A `GeneratorV0Adapter` instance that can be called with a prompt to
    generate a choice from the list of options.

    """
    warnings.warn("""
        The `choice` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Use the `Generator` object instead:
        ```python
        from outlines import Generator
        from typing import Literal
        generator = Generator(model, Literal['foo', 'bar'])
        ```
        You can then call the generator created with a prompt to generate
        a choice from the list of options.
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, OpenAI):
        if isinstance(choices, list):
            return choice_openai(model, choices, sampler)
        raise ValueError(
            "The `choice` function with OpenAI only supports a list of "
            + "strings as choices."
        )

    if isinstance(choices, type(Enum)):
        regex_str = build_regex_from_schema(pyjson.dumps(get_schema_from_enum(choices)))
    else:
        choices = [re.escape(choice) for choice in choices]  # type: ignore
        regex_str = r"(" + r"|".join(choices) + r")"

    # We do not want to raise the deprecation warning through the regex function
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
        )
        generator = regex(model, regex_str, sampler)

    if isinstance(choices, type(Enum)):
        generator.format_sequence = lambda x: pyjson.loads(x)
    else:
        generator.format_sequence = lambda x: x

    return generator


def choice_openai(
    model,
    choices: List[str],
    sampler: Sampler = multinomial(),
) -> Callable:

    choices_schema = pyjson.dumps(
        {
            "type": "object",
            "properties": {"result": {"type": "string", "enum": choices}},
            "additionalProperties": False,
            "required": ["result"],
        }
    )

    # We do not want to raise the deprecation warning through the json function
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning
        )
        generator = json(model, choices_schema, sampler)

    def generate_choice(*args, **kwargs): # pragma: no cover
        return generator(*args, **kwargs)["result"]

    return generate_choice
