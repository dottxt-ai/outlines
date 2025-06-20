---
title: Installation
---

# Installation

## Dependency Management

We recommend using modern Python packaging tools such as `uv` for managing python dependencies.

### uv (Recommended)

```shell
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install Outlines
uv venv
source .venv/bin/activate
uv pip install outlines
```

or with pip:

```shell
pip install outlines
```





## Optional Dependencies

To use Outlines models, you need to install the Python libraries for the associated inference engines/clients. Such libraries are not part of the general installation as you should only install the libraries needed for the specific models you want to use.

Outlines models with the installation of their associated additional depencies:

- [Anthropic](features/models/anthropic.md): `pip install anthropic`
- [Dottxt](features/models/dottxt.md): `pip install dottxt`
- [Gemini](features/models/gemini.md): `pip install google-generativeai`
- [Llamacpp](features/models/llamacpp.md): `pip install llama-cpp-python`
- [Mlx-lm](features/models/mlxlm.md): `pip install mlx mlx-lm`
- [Ollama](features/models/ollama.md): `pip install ollama` (after having downloaded Ollama in your system)
- [OpenAI](features/models/openai.md): `pip install openai`
- [SGLang](features/models/sglang.md): `pip install openai`
- [TGI](features/models/tgi.md): `pip install huggingface_hub`
- [Transformers](features/models/transformers.md): `pip install transformers`
- [TransformersMultiModal](features/models/transformers_multimodal.md): `pip install transformers`
- [vLLM (online server)](features/models/vllm.md): `pip install openai`
- [vLLM (offline)](features/models/vllm_offline.md): `pip install vllm`

If you encounter any problems using Outlines with these libraries, take a look at their installation instructions. The installation of `openai` and `transformers` should be straightforward, but other libraries have specific hardware requirements.

!!! warning "Hardware Requirements"

    If you are using a local model, your model may require specific hardware. Please check the documentation for these libraries.

    Some libraries like `vllm` and `llama-cpp-python` require specific hardware, such as a compatible GPU. `mlx-lm` on its side is designed for Apple Silicon, so it may not be appropriate for your use case if you are on a different platform.

## Bleeding Edge

You can install the latest version of Outlines from the repository's `main` branch:

```sh
pip install git+https://github.com/dottxt-ai/outlines.git@main
```

This can be useful, for instance, when a fix has been merged but not yet released.

## Installing for Development

See the [contributing documentation](community/contribute.md) for instructions on how to install Outlines for development, including an example using the `dot-install` method for one of the backends.
