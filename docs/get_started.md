---
title: Get Started
icon: material/human-greeting
---

# Getting started

## 1. Installation

Outlines is available on PyPi:

```bash
pip install outlines
```


!!! info "Model integrations"

    The following model integrations are available. To use them you must install the required dependencies:

    - `openai` for OpenAI models;
    - `transformers` for Hugging Face models;
    - `autoawq` for AWQ models;
    - `auto-gptq` for GPTQ models.


## 2. Hello, World

A very simple Outlines program looks like:

=== "Code"

    ```python
    import outlines

    model = outlines.models.transformers("gpt2")
    generator = outlines.generate.format(model, int)

    generate("2+2=")
    ```

=== "Output"

    ```bash
    4
    ```

The program goes through the following steps:

1. Initialize the model using the `transformers` library. Weights are loaded in memory;
2. Initialize the generator. `outlines.generate.format` constraints the output of the model
   to be a valid Python data type.
3. Call the generator with a prompt.

## 3. Going further

If you need more inspiration you can take a look at the [Examples](examples/index.md). If you have any question, or requests for documentation please reach out to us on [GitHub](https://github.com/outlines-dev/outlines/discussions), [Twitter](https://twitter.com/remilouf) or [Discord](https://discord.gg/UppQmhEpe8).

## 4. Acknowledgements

<figure markdown>
  <a href="https://www.normalcomputing.ai">
  ![Normal Computing logo](assets/images/normal_computing.jpg){ width="150" }
  </a>
</figure>

Outlines was originally developed at [@NormalComputing](https://twitter.com/NormalComputing) by [@remilouf](https://twitter.com/remilouf) and [@BrandonTWillard](https://twitter.com/BrandonTWillard).
