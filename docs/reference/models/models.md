---
title: Models
---

# Models

This section provides detailed information about using structured generation with a number of inference engines.

## Supported providers

### Open Source models

Open source models offer more flexibility for structured generation as we have control over the sampling loop:

- [transformers](transformers.md) - Run open-source models locally.
- [llama-cpp-python](llamacpp.md) - Python bindings for llama.cpp.
- [mlx-lm](mlxlm.md) - Run open-source models on Metal hardware.
- [vllm](vlm.md) - Run open-source models on the vLLM engine.
- [exllamaV2](exllamav2.md)

Ollama only supports Json Schema:

- [ollama](ollama.md) - Python client library for Ollama.

### Cloud AI providers

OpenAI has recently integrated [structured outputs][structured-outputs] in its API, and only JSON Schema-based structured generation is available. Google's Gemini API supports both Json Schema and multiple choices:

- [OpenAI](openai.md) - GPT-4o, o1, o3-mini and other OpenAI models.
- [Azure OpenAI](openai.md) - Microsoft's Azure-hosted OpenAI models.
- [Gemini](gemini.md) - Run Google's Gemini model.


## Structure generation coverage

Integrations differ in their coverage of structured generation. Here is a summary:


|                   | [Transformers](transformers.md) | [vLLM](vllm.md) | [llama.cpp](llamacpp.md) | [ExLlamaV2](exllamav2.md) | [MLXLM](mlxlm.md) | [OpenAI](openai.md) | [Gemini](gemini.md)
|-------------------|--------------|------|-----------|-----------|-------|---------|-------|
| **Supported HW**  |              |      |           |           |       |         |       |
| CUDA              | ✅           | ✅  | ✅        | ✅        | ❌    | N/A     | N/A   |
| Apple Silicon     | ✅           | ❌  | ✅        | ✅        | ✅    | N/A     | N/A   |
| x86 / AMD64       | ✅           | ❌  | ✅        | ✅        | ❌    | N/A     | N/A   |
| **Structure**     |               |     |           |           |       |         |       |
| JSON Schema       | ✅           | ✅  | ✅        | ✅        | ✅    | ✅      | ✅   |
| Choice            | ✅           | ✅  | ✅        | ✅        | ✅    | ❌      | ✅   |
| Regex             | ✅           | ✅  | ✅        | ✅        | ✅    | ❌      | ❌   |
| Grammar           | ✅           | ✅  | ✅        | ✅        | ✅    | ❌      | ❌   |


[structured-outputs]: https://platform.openai.com/docs/guides/structured-outputs
