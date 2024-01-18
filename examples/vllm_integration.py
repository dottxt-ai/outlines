import vllm
import vllm.model_executor.layers.sampler as sampler
from pydantic import BaseModel

from outlines.serve.vllm import JSONLogitsProcessor, _patched_apply_logits_processors

# Patch the _apply_logits_processors so it is compatible with `JSONLogitsProcessor`
sampler._apply_logits_processors = _patched_apply_logits_processors


class User(BaseModel):
    id: int
    name: str


llm = vllm.LLM(model="gpt2")
logits_processor = JSONLogitsProcessor(User, llm.llm_engine)
outputs = llm.generate(
    ["A prompt", "Another prompt"],
    sampling_params=vllm.SamplingParams(
        max_tokens=100, logits_processors=[logits_processor]
    ),
)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
