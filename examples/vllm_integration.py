"""Example of integrating `outlines` with `vllm`."""

import vllm
from pydantic import BaseModel

from outlines.integrations.vllm import JSONLogitsProcessor


class Person(BaseModel):
    first_name: str
    surname: str


llm = vllm.LLM(model="mistralai/Mistral-7B-v0.1", max_model_len=512)
logits_processor = JSONLogitsProcessor(schema=Person, llm=llm, whitespace_pattern=r" ?")
result = llm.generate(
    ["He is Tom Jones", "She saw Linda Smith"],
    sampling_params=vllm.SamplingParams(
        temperature=0.0,
        max_tokens=50,
        logits_processors=[logits_processor],
    ),
)
print(result)
