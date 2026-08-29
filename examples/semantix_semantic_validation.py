"""Compose outlines structural decoding with semantix-ai semantic validation.

``outlines`` constrains the *shape* of an LLM output — the generated text is
guaranteed to parse against a JSON schema. ``semantix-ai`` validates the
*meaning* of any string field against a natural-language intent, scored
locally by a small NLI cross-encoder (no API key required).

The two layers are complementary:

* ``outlines`` guarantees the response parses;
* ``semantix`` guarantees it means what we asked for.

A schema check cannot tell the difference between a polite reply and a rude
one — they both parse. This example generates a structured customer-support
reply with outlines, then checks the tone of the ``reply`` field with
semantix. We then score a deliberately rude counterexample with the same
judge to show how semantix catches a semantic violation a schema check
cannot see.

Install
-------
``pip install outlines openai "semantix-ai[nli]"``

References
----------
* semantix-ai: https://github.com/labrat-akhona/semantix-ai
"""

from enum import Enum

import openai
from pydantic import BaseModel
from semantix import NLIJudge

import outlines
from outlines import Generator
from outlines.types import JsonSchema


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    OTHER = "other"


class SupportReply(BaseModel):
    category: Category
    reply: str


POLITE_INTENT = (
    "The response is polite, professional, and acknowledges the customer's "
    "issue before offering a next step."
)


model = outlines.from_openai(openai.OpenAI(), "gpt-4o-mini")
generator = Generator(model, JsonSchema(SupportReply.model_json_schema()))

# Local cross-encoder, ~85 MB on first load, no API key.
judge = NLIJudge()

complaint = "My subscription was charged twice this month and I want a refund."

prompt = (
    "You are a customer support agent. Reply to the following complaint in "
    "JSON matching the SupportReply schema.\n\n"
    f"Complaint: {complaint}"
)

raw = generator(prompt)
result = SupportReply.model_validate_json(raw)
verdict = judge.evaluate(result.reply, POLITE_INTENT, threshold=0.5)

print("Generated reply")
print("---------------")
print(f"Category : {result.category.value}")
print(f"Reply    : {result.reply}")
print(f"Semantic : score={verdict.score:.3f}  passed={verdict.passed}")

# Counterexample. Same schema, different semantics — outlines would accept
# this (it parses); semantix flags it because the *meaning* violates the
# intent. This is the failure mode a structural validator cannot catch.
rude = SupportReply(
    category=Category.BILLING,
    reply="Read the FAQ. Not my problem.",
)
rude_verdict = judge.evaluate(rude.reply, POLITE_INTENT, threshold=0.5)

print()
print("Counterexample (schema-valid, semantically wrong)")
print("-------------------------------------------------")
print(f"Reply    : {rude.reply}")
print(f"Semantic : score={rude_verdict.score:.3f}  passed={rude_verdict.passed}")
