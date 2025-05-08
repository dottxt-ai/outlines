---
title: Installation
---

# Installation

You can install Outlines with `pip`:

```sh
pip install outlines
```

Outlines supports OpenAI, Transformers, Mamba and llama.cpp, but **you will need to install them manually**:

```sh
pip install openai
pip install transformers datasets accelerate torch
pip install llama-cpp-python
pip install transformers torch
pip install mamba_ssm transformers torch
pip install vllm
```

If you encounter any problems using Outlines with these libraries, take a look at their installation instructions. The installation of `openai` and `transformers` should be straightforward, but other libraries have specific hardware requirements.

## Optional Dependencies

Outlines provides multiple optional dependency sets to support different backends and use cases. You can install them as needed using:

- `pip install "outlines[vllm]"` for [vLLM](https://github.com/vllm-project/vllm), optimized for high-throughput inference.
- `pip install "outlines[transformers]"` for [Hugging Face Transformers](https://huggingface.co/docs/transformers/index).
- `pip install "outlines[mlx]"` for [MLX-LM](https://github.com/ml-explore/mlx-lm), optimized for Apple silicon.
- `pip install "outlines[openai]"` to use OpenAI’s API.
- `pip install "outlines[llamacpp]"` for [llama.cpp](https://github.com/ggerganov/llama.cpp), a lightweight LLM inference engine.

## Bleeding Edge

You can install the latest version of Outlines from the repository's `main` branch:

```sh
pip install git+https://github.com/dottxt-ai/outlines.git@main
```

This can be useful, for instance, when a fix has been merged but not yet released.

## Installing for Development

See the [contributing documentation](community/contribute.md) for instructions on how to install Outlines for development, including an example using the `dot-install` method for one of the backends.
