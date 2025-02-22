# mlx-lm

Outlines provides an integration with [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms), allowing models to be run quickly on Apple Silicon via the [mlx](https://ml-explore.github.io/mlx/build/html/index.html) library.

!!! Note "Installation"

    You need to install the `mlx` and `mlx-lm` libraries on a device which [supports Metal](https://support.apple.com/en-us/102894) to use the mlx-lm integration: `pip install mlx mlx-lm`.

    Consult the [`mlx-lm` documentation](https://github.com/ml-explore/mlx-examples/tree/main/llms) for detailed informations about how to initialize OpenAI clients and the available options.


## Load the model

You can initialize the model by passing the name of the repository on the HuggingFace Hub. The official repository for mlx-lm supported models is [mlx-community](https://huggingface.co/mlx-community).

```python
from mlx_lm import load
import outlines

model = outlines.from_mlxlm(*load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"))
```

This will download the model files to the hub cache folder and load the weights in memory.

## Generate text

You may generate text using the parameters described in the [text generation documentation](../text.md).

With the loaded model, you can generate text or perform structured generation, e.g.

```python
from mlx_lm import load
import outlines

model = outlines.from_mlxlm(*load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"))
answer = model("A prompt", temperature=2.0)
```

## Streaming

You may creating a streaming iterable with minimal changes

```python
from mlx_lm import load
import outlines

model = outlines.from_mlxlm(*load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"))
generator = outlines.Generator(model)

for token_str in generator.text("A prompt", temperature=2.0):
    print(token_str)
```

## Structured

You may perform structured generation with mlxlm to guarantee your output will match a regex pattern, json schema, or lark grammar.

Example: Phone number generation with pattern `"\\+?[1-9][0-9]{7,14}"`:

```python
from mlx_lm import load
import outlines

model = outlines.from_mlxlm(*load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"))

phone_number_pattern = "\\+?[1-9][0-9]{7,14}"
model_output = model("What's Jennys Number?\n", outlines.Regex(phone_number_pattern))
print(model_output)
# '8675309'
```
