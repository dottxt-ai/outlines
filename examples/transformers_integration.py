"""Example of integrating `outlines` with `transformers`."""

from pydantic import BaseModel
from transformers import pipeline

from outlines.integrations.transformers import JSONPrefixAllowedTokens


class Person(BaseModel):
    first_name: str
    surname: str


pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
prefix_allowed_tokens_fn = JSONPrefixAllowedTokens(
    schema=Person, tokenizer_or_pipe=pipe, whitespace_pattern=r" ?"
)
results = pipe(
    ["He is Tom Jones", "She saw Linda Smith"],
    return_full_text=False,
    do_sample=False,
    max_new_tokens=50,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
print(results)
