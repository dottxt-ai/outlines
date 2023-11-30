import json as pyjson
from typing import Callable, List, Optional, Union

from pydantic import BaseModel

from outlines.generate.generator import SequenceGenerator
from outlines.generate.samplers import Sampler, multinomial
from outlines.index.index import RegexFSM, StopAtTokenFSM
from outlines.index.json_schema import (
    build_regex_from_object,
    get_schema_from_signature,
)
from outlines.index.types import python_types_to_regex


def text(model, max_tokens: Optional[int] = None, *, sampler: Sampler = multinomial):
    eos_token = model.tokenizer.eos_token_id
    fsm = StopAtTokenFSM(eos_token, max_tokens)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


def regex(
    model,
    regex_str: str,
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial,
):
    fsm = RegexFSM(regex_str, model.tokenizer, max_tokens)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


def format(
    model, python_type, max_tokens: Optional[int] = None, sampler: Sampler = multinomial
):
    regex_str = python_types_to_regex(python_type)
    return regex(model, regex_str, max_tokens, sampler)


def choice(
    model,
    choices: List[str],
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial,
):
    regex_str = r"(" + r"|".join(choices) + r")"
    return regex(model, regex_str, max_tokens, sampler)


def json(
    model,
    schema_object: Union[str, object, Callable],
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial,
):
    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)

    return generator
