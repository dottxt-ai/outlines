### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings

from outlines.models.llamacpp import LlamaCpp
from outlines.models.openai import OpenAI
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.samplers import Sampler, multinomial
from outlines.v0_legacy.models.transformers_vision import TransformersVision
from outlines.types.dsl import CFG


def cfg(
    model, cfg_str: str, sampler: Sampler = multinomial()
) -> GeneratorV0Adapter:
    """Generate text in the language of a Context-Free Grammar

    This function is deprecated starting from v1.0.0. Do not use it.
    Support for it will be removed in v1.1.0.
    Use the `Generator` object instead:

    ```python
    from outlines import Generator
    from outlines.types import CFG
    cfg_str = "..."  # Context-Free Grammar as a string
    generator = Generator(model, CFG(cfg_str))
    ```

    You can then call the generator created with a prompt to generate text
    that matches the Context-Free Grammar.

    Arguments
    ---------
    model:
        The Outlines v0 model to use to generate text.
    cfg_str:
        The Context-Free Grammar to use to generate text.
    sampler:
        The sampler defining the sampling parameters.

    Returns
    -------
    A `GeneratorV0Adapter` instance that can be called with a prompt to
    generate text.

    """

    warnings.warn("""
        The `cfg` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Use the `Generator` object instead:
        ```python
        from outlines import Generator
        from outlines.types import CFG
        cfg_str = '...'  # Context-Free Grammar as a string
        generator = Generator(model, CFG(cfg_str))
        ```
        You can then call the generator created with a prompt to generate
        text in the language of a Context-Free Grammar.
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, OpenAI):
        raise NotImplementedError(
            "Cannot use CFG-based structured generation with an OpenAI model"
            + "due to the limitations of the OpenAI API."
        )
    elif isinstance(model, LlamaCpp):
        raise NotImplementedError(
            "Not yet available due to bug in llama_cpp tokenizer"
        )
    elif isinstance(model, TransformersVision):
        return GeneratorVisionV0Adapter(model, CFG(cfg_str), sampler)
    return GeneratorV0Adapter(model, CFG(cfg_str), sampler)
