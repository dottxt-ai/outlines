import warnings
from typing import Callable, List, Optional, Union

import outlines
from outlines.generate.samplers import Sampler, multinomial


def json(
    model,
    schema_object: Union[str, object, Callable],
    max_tokens: Optional[int] = None,
    *,
    sampler: Sampler = multinomial,
):
    warnings.warn(
        "`outlines.text.generate.json` is deprecated, please use `outlines.generate.json` instead. "
        "The old import path will be removed in Outlines v0.1.0.",
        DeprecationWarning,
    )
    return outlines.generate.json(model, schema_object, max_tokens, sampler=sampler)


def regex(
    model,
    regex_str: str,
    max_tokens: Optional[int] = None,
    *,
    sampler: Sampler = multinomial,
):
    warnings.warn(
        "`outlines.text.generate.regex` is deprecated, please use `outlines.generate.regex` instead. "
        "The old import path will be removed in Outlines v0.1.0.",
        DeprecationWarning,
    )
    return outlines.generate.regex(model, regex_str, max_tokens, sampler=sampler)


def format(
    model, python_type, max_tokens: Optional[int] = None, sampler: Sampler = multinomial
):
    warnings.warn(
        "`outlines.text.generate.format` is deprecated, please use `outlines.generate.format` instead. "
        "The old import path will be removed in Outlines v0.1.0.",
        DeprecationWarning,
    )
    return outlines.generate.format(model, python_type, max_tokens, sampler=sampler)


def continuation(
    model,
    max_tokens: Optional[int] = None,
    *,
    sampler: Sampler = multinomial,
    stop: Optional[Union[str, List[str]]] = None,
):
    warnings.warn(
        "`outlines.text.generate.continuation` is deprecated, please use `outlines.generate.text` instead. "
        "The old import path will be removed in Outlines v0.1.0.",
        DeprecationWarning,
    )
    if stop is not None:
        raise NotImplementedError(
            "The `stop` keyword is unavailable in the updated API. Please open an issue "
            " at https://github.com/outlines-dev/outlines/issues if you need it implemented."
        )

    return outlines.generate.text(model, max_tokens, sampler=sampler)


def choice(
    model,
    choices: List[str],
    max_tokens: Optional[int] = None,
    *,
    sampler: Sampler = multinomial,
):
    warnings.warn(
        "`outlines.text.generate.choice` is deprecated, please use `outlines.generate.choice` instead. "
        "The old import path will be removed in Outlines v0.1.0.",
        DeprecationWarning,
    )
    return outlines.generate.choice(model, choices, max_tokens, sampler)
