# mlx-lm

Outlines provides an integration with [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms), allowing models to be run quickly on Apple Silicon via the [mlx](https://ml-explore.github.io/mlx/build/html/index.html) library.

!!! Note "Installation"

    You need to install the `mlx` and `mlx-lm` libraries on a device which [supports Metal](https://support.apple.com/en-us/102894) to use the mlx-lm integration. To get started quickly you can also run:

    ```bash
    pip install "outlines[mlxlm]"
    ```


## Load the model

You can initialize the model by passing the name of the repository on the HuggingFace Hub. The official repository for mlx-lm supported models is [mlx-community](https://huggingface.co/mlx-community).

```python
from outlines import models

model = models.mlxlm("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")
```

This will download the model files to the hub cache folder and load the weights in memory.

The arguments `model_config` and `tokenizer_config` are available to modify loading behavior. For example, per the `mlx-lm` [documentation](https://github.com/ml-explore/mlx-examples/tree/main/llms#supported-models), you must set an eos_token for `qwen/Qwen-7B`. In outlines you may do so via

```
model = models.mlxlm(
    "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)
```

**Main parameters:**

(Subject to change. Table based on [mlx-lm.load docstring](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py#L429))

| Parameters         | Type   | Description                                                                                      | Default |
|--------------------|--------|--------------------------------------------------------------------------------------------------|---------|
| `tokenizer_config` | `dict` | Configuration parameters specifically for the tokenizer. Defaults to an empty dictionary.        | `{}`    |
| `model_config`     | `dict` | Configuration parameters specifically for the model. Defaults to an empty dictionary.            | `{}`    |
| `adapter_path`     | `str`  | Path to the LoRA adapters. If provided, applies LoRA layers to the model.                        | `None`  |
| `lazy`             | `bool` | If False, evaluate the model parameters to make sure they are loaded in memory before returning. | `False` |


## Generate text

You may generate text using the parameters described in the [text generation documentation](../text.md).

With the loaded model, you can generate text or perform structured generation, e.g.

```python
from outlines import models, generate

model = models.mlxlm("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")
generator = generate.text(model)

answer = generator("A prompt", temperature=2.0)
```

## Streaming

You may creating a streaming iterable with minimal changes

```python
from outlines import models, generate

model = models.mlxlm("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")
generator = generate.text(model)

for token_str in generator.text("A prompt", temperature=2.0):
    print(token_str)
```

## Structured

You may perform structured generation with mlxlm to guarantee your output will match a regex pattern, json schema, or lark grammar.

Example: Phone number generation with pattern `"\\+?[1-9][0-9]{7,14}"`:

```python
from outlines import models, generate

model = models.mlxlm("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")

phone_number_pattern = "\\+?[1-9][0-9]{7,14}"
generator = generate.regex(model, phone_number_pattern)

model_output = generator("What's Jennys Number?\n")
print(model_output)
# '8675309'
```
