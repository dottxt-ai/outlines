# OpenAI

!!! Installation

    You need to install the `openai` library to be able to use the OpenAI API in Outlines. Install all optional dependencies of the `OpenAI` model with: `pip install outlines[openai]`.

    You also need to have an OpenAI API key. This API key must either be set as an environment variable called `OPENAI_API_KEY` or be provided to the `openai.OpenAI` class when instantiating it.

## Model Initialization

To create an OpenAI model instance, you can use the `from_openai` function. It takes 2 arguments:

- `client`: an `openai.OpenAI` or `openai.AzureOpenAI` instance
- `model_name`: the name of the model you want to use

For instance:

```python
import outlines
import openai

# Create the client
client = openai.OpenAI()

# Create the model
model = outlines.from_openai(
    client,
    "gpt-4o"
)
```

Check the [OpenAI documentation](https://platform.openai.com/docs/models) for an up-to-date list of available models. As shown above, you can use Azure OpenAI in Outlines the same way you would use OpenAI, just provide an `openai.AzureOpenAI` instance to the Outlines model class.

## Text Generation

Once you've created your Outlines `OpenAI` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt.

For instance:

```python
import openai
import outlines

# Create the model
model = outlines.from_openai(
    openai.OpenAI(),
    "gpt-4o"
)

# Call it to generate text
response = model("What's the capital of Latvia?", max_tokens=20)
print(response) # 'Riga'
```

#### Vision

Some OpenAI models support vision input. To use this feature, provide a list containing a text prompt and `Image` instances.

For instance:

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

#### Chat

You can also use chat inputs with the `OpenAI` model. To do so, call the model with a `Chat` instance. The content of messsage within the chat can be vision inputs as described above.

For instance:

```python
import io
import requests
import PIL
import openai
import outlines
from outlines.inputs import Chat, Image

# Create the model
model = outlines.from_openai(
    openai.OpenAI(),
    "gpt-4o"
)

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

# Call the model to generate a response
response = model(prompt, max_tokens=50)
print(response) # 'This is a picture of a black dog.'
```

#### Streaming

Finally, the `OpenAI` model supports streaming through the `stream` method.

For instance:

```python
import openai
import outlines

# Create the model
model = outlines.from_openai(
    openai.OpenAI(),
    "gpt-4o"
)

# Stream the response
for chunk in model.stream("Tell me a short story about a cat.", max_tokens=50):
    print(chunk) # 'Once...'
```

## Structured Generation

OpenAI provides supports for some forms of structured output: JSON schemas and JSON syntax. To use it, call the model with an `output_type` on top of your prompt.

#### JSON Schema

```python
from typing import List
from pydantic import BaseModel
import openai
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_openai(openai.OpenAI(), "gpt-4o")

# Call it with the output type to generate structured text
result = model("Create a character, use the json format.", Character, top_p=0.1)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

#### JSON Syntax

What we mean by JSON syntax is what is sometimes called JSON mode, meaning that the model will return a valid JSON, but you do not get to specify its structure. To use this JSON mode, provide the `dict` type as an output type.

```python
import openai
import outlines

# Create the model
model = outlines.from_openai(openai.OpenAI(), "gpt-4o")

# Call it with the output type to generate structured text
result = model("Create a character, use the json format.", dict, temperature=0.5)
print(result) # '{"first_name": "Henri", "last_name": "Smith", "height": "170"}'
```

## Inference arguments

When calling the model, you can provide keyword arguments that will be passed down to the `chat.completions.create` method of the OpenAI client. Some of the most common arguments include `max_tokens`, `temperature`, `stop` and `top_p`.

Another keyword argument of interest is `n`. If set with an integer value superior to 1, OpenAI will generate several sample responses and you will receive a list of strings as a response to your model call.

See the [OpenAI API documentation](http://platform.openai.com/docs/api-reference/chat/create) for the full list of available arguments.
