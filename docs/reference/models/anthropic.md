# Anthropic

!!! Installation

    You need to install the `anthropic` library to be able to use the Anthropic API in Outlines: `pip install anthropic`.

## Anthropic models

Outlines supports models available via the Anthropic API, e.g. Claude 3.5 Haiku or Claude 3.5 Sonner. You can initialize the model by passing the model name to `outlines.models.Anthropic`:

```python
from anthropic import Anthropic
import outlines

model = outlines.from_anthropic(Anthropic("claude-3-5-haiku-latest"))
model = outlines.from_anthropic(Anthropic("claude-3-5-sonnet-latest"))
```

Check the [Anthropic documentation](https://docs.anthropic.com/en/docs/about-claude/models) for an up-to-date list of available models.

## Text generation

To generate text using an Anthropic model you need to build a `Generator` object, possibly with the desired output type. You can then call the model by calling the `Generator`. The method accepts every argument that you could pass to the `client.completions.create` function, as keyword arguments:

```python
from anthropic import Anthropic
import outlines

model = outlines.from_anthropic(Anthropic("claude-3-5-haiku-latest"))
generator = Generator(model)
result = generator("Prompt", max_tokens=1024)

# Call the model directly
result = model("Prompt", max_tokens=1024)
```


!!! Warning

    You must set a value for `max_tokens` with Anthropic models.

See the [Anthropic SDK documentation](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/resources/messages.py) for the list of available arguments.

The Anthropic API currently does not support structured generation.
