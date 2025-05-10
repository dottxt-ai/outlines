---
title: Installation
---

# Installation

## Dependency Management

We recommend using modern Python packaging tools such as `uv` for managing python dependencies.

### uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install Outlines
uv venv
source .venv/bin/activate
uv pip install outlines
```

or with basic pip:

```bash
pip install outlines
```

Outlines supports the following model types:

- Transformers
- OpenAI
- llama.cpp
- vLLM
- dottxt
- Anthropic
- Ollama
- mlx-lm
- Google

To use these model providers, you must install them manually, i.e.

```sh
pip install openai
pip install transformers datasets accelerate torch
pip install llama-cpp-python
pip install transformers torch
pip install vllm
pip install dottxt
pip install anthropic
pip install ollama
pip install mlx-lm
pip install google-generativeai
```

If you encounter any problems using Outlines with these libraries, take a look at their installation instructions. The installation of `openai` and `transformers` should be straightforward, but other libraries have specific hardware requirements.

!!! warning "Hardware Requirements"

    If you are using an offline inference tool (not a remote server), your model may require specific hardware. Please check the documentation for these libraries.

    Some libraries like `vllm` and `llama-cpp-python` require specific hardware, such as a compatible GPU. `mlx-lm` is designed for Apple Silicon, so may not be appropriate for your use case if you are on a different platform.

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
