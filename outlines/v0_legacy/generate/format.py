### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings

from outlines.models.openai import OpenAI
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.models.transformers_vision import TransformersVision
from outlines.v0_legacy.samplers import Sampler, multinomial


def format(
    model, python_type, sampler: Sampler = multinomial()
) -> GeneratorV0Adapter:
    """Generate structured data that can be parsed as a Python type.

    This function is deprecated starting from v1.0.0. Do not use it.
    Support for it will be removed in v1.1.0.
    Use the `Generator` object instead::

    ```python
    from outlines import Generator
    generator = Generator(model, int)
    ```

    You can then call the generator created with a prompt to generate a
    integer.

    Parameters
    ----------
    model:
        The Outlines v0 model to use to generate text.
    python_type:
        A Python type. The output of the generator must be parseable into
        this type.
    sampler:
        The sampler defining the sampling parameters.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the Python type
    and translates this text into the corresponding type.

    """
    warnings.warn("""
        The `format` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Use the `Generator` object instead:
        ```python
        from outlines import Generator
        generator = Generator(model, int)
        ```
        You can then call the generator created with a prompt to generate
        an integer.
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, OpenAI):
        raise NotImplementedError(
            "Cannot use Python type-structured generation with an OpenAI model"
            + " due to the limitations of the OpenAI API."
        )
    elif isinstance(model, TransformersVision):
        return GeneratorVisionV0Adapter(model, python_type, sampler)
    return GeneratorV0Adapter(model, python_type, sampler)
