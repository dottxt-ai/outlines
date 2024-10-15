---
title: installation
---

# Installation

You can install Outlines with `pip`:

```python
pip install outlines
```

Outlines supports OpenAI, transformers, Mamba, llama.cpp and exllama2 but **you will need to install them manually**:

```python
pip install openai
pip install transformers datasets accelerate torch
pip install llama-cpp-python
pip install exllamav2 transformers torch
pip install mamba_ssm transformers torch
pip install vllm
```

If you encounter any problem using Outlines with these libraries, take a look at their installation instructions. The installation of `openai` and `transformers` should be straightforward, but other libraries have specific hardware requirements.

## Bleeding edge

You can install the latest version of Outlines on the repository's `main` branch:

```python
pip install git+https://github.com/dottxt-ai/outlines.git@main
```

This can be useful, for instance, when a fix has been merged but not yet released.

## Installing for development

See the [contributing documentation](community/contribute.md) for instructions on how to install Outlines for development.
