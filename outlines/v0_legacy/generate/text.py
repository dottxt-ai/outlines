### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings
from typing import Union

from outlines.models.openai import OpenAI
from outlines.v0_legacy.models.transformers_vision import TransformersVision
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.samplers import Sampler, multinomial


def text(model, sampler: Sampler = multinomial()) -> Union[
    GeneratorV0Adapter, GeneratorVisionV0Adapter, OpenAI
]:
    """Generate unconstrained text.

    This function is deprecated starting from v1.0.0. Do not use it.
    Support for it will be removed in v1.1.0.
    Use the `Generator` object instead:

    ```python
    from outlines import Generator
    generator = Generator(model)
    ```

    You can then call the generator created with a prompt to generate text.

    Parameters
    ----------
    model:
        The Outlines v0 model to use to generate text.
    sampler:
        The sampler defining the sampling parameters.

    Returns
    -------
    A `GeneratorV0Adapter` instance that can be called with a prompt to
    generate text.

    """
    warnings.warn("""
        The `text` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Use the `Generator` object instead:
        ```python
        from outlines import Generator
        generator = Generator(model)
        ```
        You can then call the generator created with a prompt to generate
        text.
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, OpenAI):
        return model
    if isinstance(model, TransformersVision):
        return GeneratorVisionV0Adapter(model, None, sampler)
    return GeneratorV0Adapter(model, None, sampler)
