---
title: installation
---

# Installation

You can install Outlines with `pip`:

```python
pip install outlines
```

Outlines supports OpenAI, transformers, Mamba, AutoGPTQ, AutoAWQ but **you will need to install them manually**:

```python
pip install openai
pip install transformers datasets accelerate
pip install autoawq
pip install auto-gptq
pip install mamba_ssm
```

If you encounter any problem using Outlines with these libraries, take a look at their installation instructions. The installation of `openai` and `transformers` should be straightforward, but other libraries have specific hardware requirements. We summarize them below:

### AutoGPTQ

- `pip install auto-gptq` works with CUDA 12.1
- For CUDA 11.8, see the [documentation](https://github.com/PanQiWei/AutoGPTQ?tab=readme-ov-file#installation)
- `pip install auto-gptq[triton]` to use the Triton backend

Still encounter an issue? See the [documentation](https://github.com/PanQiWei/AutoGPTQ?tab=readme-ov-file#installation) for up-to-date information.


### AutoAWQ

- Your GPU(s) must be of Compute Capability 7.5. Turing and later architectures are supported.
- Your CUDA version must be CUDA 11.8 or later.

Still encounter an issue? See the [documentation](https://github.com/casper-hansen/AutoAWQ?tab=readme-ov-file#install) for up-to-date information.


## Installing for development

See the [contributing documentation](community/contribute.md) for instructions on how to install Outlines for development.
