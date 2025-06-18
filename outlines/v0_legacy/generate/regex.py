### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings

from outlines.models.openai import OpenAI
from outlines.types import Regex
from outlines.v0_legacy.generate.api import GeneratorV0Adapter, GeneratorVisionV0Adapter
from outlines.v0_legacy.models.transformers_vision import TransformersVision
from outlines.v0_legacy.samplers import Sampler, multinomial


def regex(model, regex_str: str | Regex, sampler: Sampler = multinomial()):
    """Generate structured text in the language of a regular expression.

    This function is deprecated starting from v1.0.0. Do not use it.
    Support for it will be removed in v1.1.0.
    Use the `Generator` object instead:

    ```python
    from outlines import Generator
    from outlines.types import Regex
    generator = Generator(model, Regex(r"^[a-z]+$"))
    ```

    You can then call the generator created with a prompt to generate text
    that matches the regular expression.

    Parameters
    ----------
    model:
        The Outlines v0 model to use to generate text.
    regex_str:
        The regular expression that the output must follow.
    sampler:
        The sampler defining the sampling parameters.

    Returns
    -------
    A `GeneratorV0Adapter` instance that can be called with a prompt to
    generate text.

    """
    warnings.warn("""
        The `regex` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Use the `Generator` object instead:
        ```python
        from outlines import Generator
        from outlines.types import Regex
        generator = Generator(model, Regex(r'regex_str'))
        ```
        You can then call the generator created with a prompt to generate
        text that matches the regular expression.
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, OpenAI):
        raise NotImplementedError(
            "Cannot use regex-structured generation with an OpenAI model"
            + "due to the limitations of the OpenAI API."
        )
    if isinstance(model, TransformersVision):
        return GeneratorVisionV0Adapter(model, regex_str, sampler)
    return GeneratorV0Adapter(model, regex_str, sampler)
