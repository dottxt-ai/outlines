### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings

import interegular

from outlines.models.openai import OpenAI
from outlines.types import FSM
from outlines.v0_legacy.samplers import Sampler, multinomial
from outlines.v0_legacy.generate.api import (
    GeneratorV0Adapter,
    GeneratorVisionV0Adapter
)
from outlines.v0_legacy.models.transformers_vision import TransformersVision


def fsm(
    model, fsm: interegular.fsm.FSM, sampler: Sampler = multinomial()
) -> GeneratorV0Adapter:
    """Generate text that follows a Finite State Machine.

    This function is deprecated starting from v1.0.0. Do not use it.
    Support for it will be removed in v1.1.0.
    Use the `Generator` object instead:

    ```python
    import interegular
    from outlines import Generator
    from outlines.types import FSM
    fsm = FSM(interegular.parse_pattern(r"[a-z]+").to_fsm())
    generator = Generator(model, fsm)
    ```

    You can then call the generator created with a prompt to generate text
    that follows the Finite State Machine.

    Parameters
    ----------
    model:
        The Outlines v0 model to use to generate text.
    fsm:
        The interregular `FSM` to use to generate text.
    sampler:
        The sampler defining the sampling parameters.

    Returns
    -------
    A `GeneratorV0Adapter` instance that can be called with a prompt to
    generate text that follows the Finite State Machine.

    """
    warnings.warn("""
        The `fsm` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Use the `Generator` object instead:
        ```python
        from outlines import Generator
        from outlines.types import FSM
        fsm_str = '...'  # Finite State Machine as a string
        generator = Generator(model, FSM(fsm_str))
        ```
        You can then call the generator created with a prompt to generate
        text that follows the Finite State Machine.
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, OpenAI):
        raise NotImplementedError(
            "Cannot use FSM-based generation with an OpenAI model"
            + "due to the limitations of the OpenAI API."
        )
    if isinstance(model, TransformersVision):
        return GeneratorVisionV0Adapter(model, FSM(fsm), sampler)
    return GeneratorV0Adapter(model, FSM(fsm), sampler)
