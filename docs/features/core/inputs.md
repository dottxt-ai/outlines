---
title: Model Inputs
---

# Model Inputs

Outlines models accept various types of inputs to generate text. The input format depends on the capabilities of the underlying model and the type of task you want to perform. The most basic type of input is a single string prompt, it's accepted by all models.

## Overview

The model input is the first argument of the `__call__`, `stream` and `batch` methods of both models and generators.

There are 3 types of model inputs:

- **Text prompts** - Simple strings
- **Multimodal inputs** - List containning a string prompt along with assets
- **Chat inputs** - `Chat` instances containing messages

## Text Prompts

The simplest form of input is a plain text string. This works with all models and is suitable for standard text generation tasks.

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create a model
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

# Simple text prompt
response = model("What's the capital of France?", max_new_tokens=20)
print(response)  # 'Paris'
```

## Multimodal Inputs (Vision)

For models that support them, you can provide a list containing a text prompt and one or more assets.

There are 3 types of assets defined in Outlines:

- `Image`: contains a PIL Image
- `Video`: contains any object (you must choose a format that is supported by your model)
- `Audio`: contains any object (you must choose a format that is supported by your model)

Among those, `Image` is by far the most important as multiple models support vision inputs.

For instance with vision input:

```python
import io
import requests
import PIL
import outlines
import openai
from outlines.inputs import Image

# Create the model
model = outlines.from_openai(
    openai.OpenAI(),
    "gpt-4o"
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
print(response) # 'This is a picture of a black dog.'
```

## Chat Inputs

For conversational models, you can use the `Chat` class to provide a conversation history with multiple messages.

A `Chat` instance is instantiated with an optional list of messages. Each message must be a dictionary containing two mandatory keys:
- `role`: must be one of `system`, `assistant` or `user`
- `content`: must be either a string or a multimodal input (if the model supports it)

For instance:

```python
import io
import requests
import PIL
from outlines.inputs import Chat, Image

# Function to get an image
def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

# Create the chat input
prompt = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": ["Describe the image", Image(get_image("https://picsum.photos/id/237/400/300"))]
    },
])
print(prompt)
# {'role': 'system', 'content': 'You are a helpful assistant.'}
# {'role': 'user', 'content': ['Describe the image', Image(image=<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=400x300 at 0x7FFA526CCC40>)]}
```

After having created a `Chat` instance, you can add one or several messages thanks to the `append` and `extend` methods. You can also remove the last message of the Chat with the `pop` method.

For instance:

```python
from outlines.inputs import Chat

# Create the chat input
prompt = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
])

# Add a message
prompt.append({"role": "user", "content": "How are you doing today?"})
print(prompt)
# {'role': 'system', 'content': 'You are a helpful assistant.'}
# {'role': 'user', 'content': 'How are you doing today?'}

# Remove the last messsage
last_message = prompt.pop()
print(last_message)
# {'role': 'user', 'content': 'How are you doing today?'}
print(prompt)
# {'role': 'system', 'content': 'You are a helpful assistant.'}

# RAdd several messages
prompt.extend([
    {"role": "user", "content": "How are you doing today?"},
    {"role": "assistant", "content": "Excellent, thanks!"}
])
print(prompt)
# {'role': 'system', 'content': 'You are a helpful assistant.'}
# {'role': 'user', 'content': 'How are you doing today?'}
# {'role': 'assistant', 'content': 'Excellent, thanks!'}
```

Finally, there are three convenience method to easily add a message:

- add_system_message
- add_user_message
- add_assistant_message

As the role is already set, you only need to provide the content.

For instance:

```python
from outlines.inputs import Chat

# Create the chat input
prompt = Chat()

prompt.add_system_message("You are a helpful assistant.")
prompt.add_system_message("How are you doing today?")
prompt.add_system_message("Excellent, thanks!")

print(prompt)
# {'role': 'system', 'content': 'You are a helpful assistant.'}
# {'role': 'user', 'content': 'How are you doing today?'}
# {'role': 'assistant', 'content': 'Excellent, thanks!'}
```

## Batching

In the case of batching, for models that support it, you just have to provide several instances of the model inputs described above in a list.

For instance:

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create model
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
)

# Create a list of prompts that will be used in a single batch
prompts = [
    "What's the capital of Lithuania?",
    "What's the capital of Latvia?",
    "What's the capital of Estonia?"
]

# Call it to generate text
result = model.batch(prompts, max_new_tokens=20)
print(result) # ['Vilnius', 'Riga', 'Tallinn']
```
