import vllm
from pydantic import BaseModel

from outlines.serve.vllm import JSONLogitsProcessor


class User(BaseModel):
    id: int
    name: str


llm = vllm.LLM(model="openai-community/gpt2")
logits_processor = JSONLogitsProcessor(schema=User, llm=llm)
result = llm.generate(
    ["A prompt", "Another prompt"],
    sampling_params=vllm.SamplingParams(
        max_tokens=100, logits_processors=[logits_processor]
    ),
)
print(result)
