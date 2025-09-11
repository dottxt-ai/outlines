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

Once you've created your Outlines `Anthropic` model instance, you're all set to generate text with this provider. You can simply call the model with a text prompt.

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
print(response.content) # 'Riga'
```

#### Vision

Some Anthropic models support vision input. To use this feature, provide a list containing a text prompt and `Image` instances.

For instance:

```python
import io
import requests
import PIL
from anthropic import Anthropic
from outlines import from_anthropic
from outlines.inputs import Image

# Create the model
model = from_anthropic(
    Anthropic(),
    "claude-3-5-sonnet-latest"
)

# Function to get an image
def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# Create the prompt containing the text and the image
prompt = [
    "Describe the image",
    Image(get_image("https://picsum.photos/id/237/400/300"))
]

# Call the model to generate a response
response = model(prompt, max_tokens=50)
print(response.content) # 'This is a picture of a black dog.'
```

#### Chat

You can also use chat inputs with the `Anthropic` model. To do so, call the model with a `Chat` instance. The content of messsage within the chat can be vision inputs as described above.

For instance:

```python
import io
import requests
import PIL
from anthropic import Anthropic
from outlines import from_anthropic
from outlines.inputs import Chat, Image

# Create the model
model = from_anthropic(
    Anthropic(),
    "claude-3-5-sonnet-latest"
)

# Function to get an image
def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# Create the chat input
prompt = Chat([
    {"role": "user", "content": "You are a helpful assistant that helps me described pictures."},
    {"role": "assistant", "content": "I'd be happy to help you describe pictures! Please go ahead and share an image"},
    {
        "role": "user",
        "content": ["Describe the image", Image(get_image("https://picsum.photos/id/237/400/300"))]
    },
])

# Call the model to generate a response
response = model(prompt, max_tokens=50)
print(response.content) # 'This is a picture of a black dog.'
```

#### Streaming

Finally, the `Anthropic` model supports streaming through the `stream` method.

For instance:

```python
from anthropic import Anthropic
import outlines

# Create the model
model = outlines.from_anthropic(
    Anthropic(),
    "claude-3-5-sonnet-latest"
)

# Stream the response
for chunk in model.stream("Tell me a short story about a cat.", max_tokens=50):
    print(chunk.content) # 'Once...'
```

#### Tools

Anthropic supports tool calling. To use it, provide a list of `tools` to the model.

For instance:

```python
from anthropic import Anthropic
from outlines import from_anthropic
from outlines.inputs import Chat
from typing import Optional

# Our tool
def get_weather(city: str, hour: Optional[int] = None):
    """Give the weather for a given city, and optionally for a specific hour of the day"""
    return "20 degrees"

# Create the model
model = from_anthropic(
    Anthropic(),
    "claude-3-5-sonnet-latest"
)

# Call the model with the tool defined above
chat = Chat([
    {"role": "user", "content": "What's the weather in Tokyo?"},
])
response = model(chat, tools=[get_weather], max_tokens=100)
print(response.tool_calls) # [ToolCallOutput(name='get_weather', id='toolu_01WDUo65vCXkrmjD3Yehbc5v', args={'city': 'Tokyo'})]
```

## Inference arguments

When calling the model or streaming, you can provide keyword arguments that will be passed down to the Anthropic client. Make sure to include all the arguments you need to configure the client's behavior to your expected behavior. Some of the most common arguments include `max_tokens`, `temperature`, `stop_sequences` and `top_k`.

See the [Anthropic API documentation](https://docs.anthropic.com/en/api/messages) for the full list of available arguments.

!!! Warning

    You must set a value for `max_tokens` with Anthropic models.
