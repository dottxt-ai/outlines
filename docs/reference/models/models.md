---
title: Models
---

# Models

Outlines supports generation using a number of inference engines (`outlines.models`). Loading a model using outlines follows a similar interface between inference engines:

```python
import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-128k-instruct")
model = outlines.models.transformers_vision("llava-hf/llava-v1.6-mistral-7b-hf")
model = outlines.models.vllm("microsoft/Phi-3-mini-128k-instruct")
model = outlines.models.llamacpp(
    "microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3-mini-4k-instruct-q4.gguf"
)
model = outlines.models.exllamav2("bartowski/Phi-3-mini-128k-instruct-exl2")
model = outlines.models.mlxlm("mlx-community/Phi-3-mini-4k-instruct-4bit")

model = outlines.models.openai(
    "gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"]
)
```


# Feature Matrix
|                   | [Transformers](transformers.md) | [Transformers Vision](transformers_vision.md) | [vLLM](vllm.md) | [llama.cpp](llamacpp.md) | [ExLlamaV2](exllamav2.md) | [MLXLM](mlxlm.md) | [OpenAI](openai.md)* |
|-------------------|--------------|---------------------|------|-----------|-----------|-------|---------|
| **Device**        |              |                     |      |           |           |       |         |
| Cuda              | ✅           | ✅                  | ✅   | ✅        | ✅        | ❌    | N/A     |
| Apple Silicon     | ✅           | ✅                  | ❌   | ✅        | ✅        | ✅    | N/A     |
| x86 / AMD64       | ✅           | ✅                  | ❌   | ✅        | ✅        | ❌    | N/A     |
| **Sampling**      |              |                     |      |           |           |       |         |
| Greedy            | ✅           | ✅                  | ✅   | ✅*       | ✅        | ✅    | ❌      |
| Multinomial       | ✅           | ✅                  | ✅   | ✅        | ✅        | ✅    | ✅      |
| Multiple Samples  | ✅           | ✅                  |      | ❌        |           | ❌    | ✅      |
| Beam Search       | ✅           | ✅                  | ✅   | ❌        | ✅        | ❌    | ❌      |
| **Generation**    |              |                     |      |           |           |       |         |
| Batch             | ✅           | ✅                  | ✅   | ❌        | ?         | ❌    | ❌      |
| Stream            | ✅           | ❌                  | ❌   | ✅        | ?         | ✅    | ❌      |
| **`outlines.generate`** |        |                     |      |           |           |       |         |
| Text              | ✅           | ✅                  | ✅   | ✅        | ✅        | ✅    | ✅      |
| __Structured__    | ✅           | ✅                  | ✅   | ✅        | ✅        | ✅    | ✅      |
| JSON Schema       | ✅           | ✅                  | ✅   | ✅        | ✅        | ✅    | ✅      |
| Choice            | ✅           | ✅                  | ✅   | ✅        | ✅        | ✅    | ✅      |
| Regex             | ✅           | ✅                  | ✅   | ✅        | ✅        | ✅    | ❌      |
| Grammar           | ✅           | ✅                  | ✅   | ✅        | ✅        | ✅    | ❌      |


## Caveats

- OpenAI doesn't support structured generation due to limitations in their API and server implementation.
- `outlines.generate` ["Structured"](../generation/generation.md) includes methods such as `outlines.generate.regex`, `outlines.generate.json`, `outlines.generate.cfg`, etc.
- MLXLM only supports Apple Silicon.
- llama.cpp greedy sampling available via multinomial with `temperature = 0.0`.
