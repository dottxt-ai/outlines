import warnings
from typing import Callable, List, Optional, Union

import outlines
from outlines.samplers import MultinomialSampler, Sampler


def json(
    model,
    schema_object: Union[str, object, Callable],
    max_tokens: Optional[int] = None,
    *,
    sampler: Sampler = MultinomialSampler(),
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
    sampler: Sampler = MultinomialSampler(),
):
    warnings.warn(
        "`outlines.text.generate.regex` is deprecated, please use `outlines.generate.regex` instead. "
        "The old import path will be removed in Outlines v0.1.0.",
        DeprecationWarning,
    )
    return outlines.generate.regex(model, regex_str, max_tokens, sampler=sampler)


def format(
    model,
    python_type,
    max_tokens: Optional[int] = None,
    sampler: Sampler = MultinomialSampler(),
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
    stop_at: Optional[Union[str, List[str]]] = None,
    *,
    sampler: Sampler = MultinomialSampler(),
):
    warnings.warn(
        "`outlines.text.generate.continuation` is deprecated, please use `outlines.generate.text` instead. "
        "The old import path will be removed in Outlines v0.1.0.",
        DeprecationWarning,
    )

    return outlines.generate.text(model, max_tokens, stop_at, sampler=sampler)


def choice(
    model,
    choices: List[str],
    max_tokens: Optional[int] = None,
    *,
    sampler: Sampler = MultinomialSampler(),
):
    warnings.warn(
        "`outlines.text.generate.choice` is deprecated, please use `outlines.generate.choice` instead. "
        "The old import path will be removed in Outlines v0.1.0.",
        DeprecationWarning,
    )
    return outlines.generate.choice(model, choices, max_tokens, sampler=sampler)
