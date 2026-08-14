"""Example of integrating `outlines` with `vllm`."""

import vllm
from pydantic import BaseModel
from transformers import AutoTokenizer

from outlines.models.vllm_offline import adapt_tokenizer
from outlines.processors import JSONLogitsProcessor


class Person(BaseModel):
    first_name: str
    surname: str


MODEL_ID = "mistralai/Mistral-7B-v0.1"
llm = vllm.LLM(model=MODEL_ID, max_model_len=512)
tokenizer = adapt_tokenizer(AutoTokenizer.from_pretrained(MODEL_ID))
logits_processor = JSONLogitsProcessor(
    schema=Person,
    tokenizer=tokenizer,
    tensor_library_name="torch",
    whitespace_pattern=r" ?"
)
result = llm.generate(
    ["He is Tom Jones", "She saw Linda Smith"],
    sampling_params=vllm.SamplingParams(
        temperature=0.0,
        max_tokens=50,
        logits_processors=[logits_processor],
    ),
)
print(result)
