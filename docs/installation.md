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
pip install transformers datasets accelerate
pip install llama-cpp-python
pip install mamba_ssm
```

If you encounter any problem using Outlines with these libraries, take a look at their installation instructions. The installation of `openai` and `transformers` should be straightforward, but other libraries have specific hardware requirements.

## Installing for development

See the [contributing documentation](community/contribute.md) for instructions on how to install Outlines for development.
