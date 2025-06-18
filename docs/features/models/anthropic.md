---
title: Anthropic
---

# Anthropic

!!! Installation

    You need to install the `anthropic` library to be able to use the Anthropic API in Outlines. Install all optional dependencies of the `Anthropic` model with: `pip install outlines[anthropic]`.

    You also need to have an Anthropic API key. This API key must either be set as an environment variable called `ANTHROPIC_API_KEY` or be provided to the `anthropic.Anthropic` class when instantiating it.

## Model Initialization

To create an Anthropic model instance, you can use the `from_anthropic` function. It takes 2 arguments:

- `client`: an `anthropic.Anthropic` instance
- `model_name`: the name of the model you want to use in subsequent model calls (optional)

For instance:

```python
from anthropic import Anthropic
import outlines

# Create the Anthropic client
client = Anthropic()

# Create the model
model = outlines.from_anthropic(
    client,
    "claude-3-5-sonnet-latest"
)
```

Check the [Anthropic documentation](https://docs.anthropic.com/en/docs/about-claude/models) for an up-to-date list of available models.

## Text Generation

Once you've created your Outlines `Anthropic` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt.

For instance:

```python
from anthropic import Anthropic
import outlines

# Create the model
model = outlines.from_anthropic(
    Anthropic(),
    "claude-3-5-sonnet-latest"
)

# Call it to generate text
response = model("What's the capital of Latvia?", max_tokens=20)
print(response) # 'Riga'
```

Anthropic does not support structured generation, so you cannot provide an output type. It does support streaming and vision inputs though.

For instance:

```python
import io
import requests
import PIL
from anthropic import Anthropic
from outlines import from_anthropic
from outlines.templates import Vision

# Create the model
model = from_anthropic(
    Anthropic(),
    "claude-3-5-sonnet-latest"
)

# Function to get an image
def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# Create the prompt
prompt = Vision("Describe the image", get_image("https://picsum.photos/id/237/400/300"))

# Stream the response
for chunk in model.stream(prompt, max_tokens=50):
    print(chunk) # 'This...'
```

## Inference arguments

When calling the model or streaming, you can provide keyword arguments that will be passed down to the Anthropic client. Make sure to include all the arguments you need to configure the client's behavior to your expected behavior. Some of the most common arguments include `max_tokens`, `temperature`, `stop_sequences` and `top_k`.

See the [Anthropic API documentation](https://docs.anthropic.com/en/api/messages) for the full list of available arguments.

!!! Warning

    You must set a value for `max_tokens` with Anthropic models.
