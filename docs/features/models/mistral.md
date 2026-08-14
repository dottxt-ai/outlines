# Mistral

!!! Installation

    You need to install the `mistralai` library to be able to use the Mistral API in Outlines. Install all optional dependencies of the `Mistral` model with: `pip install "outlines[mistral]"`.

    You also need to have an Mistral API key. This API key must either be set as an environment variable called `MISTRAL_API_KEY` or be provided to the `mistralai.Mistral` class when instantiating it.

## Model Initialization

To create an `Mistral` or `AsyncMistral` model instance, you can use the `from_mistral` function. It takes 3 arguments:

- `client`: a `mistralai.Mistral` instance
- `model_name` (optional): the name of the model you want to use
- `async_client` (optional): whether it should create a sync or an async model

As the `mistralai` library uses a single class to handle both sync and async requests, you must set the `async_client` argument to True to get an `AsyncMistral` model.

For instance:

```python
import mistralai
import outlines

# Create the Mistral client
client = mistral.Mistral()

# Create a sync model
model = outlines.from_mistral(
    client,
    "mistral-large-latest"
)

# Create aa async model
model = outlines.from_mistral(
    client,
    "mistral-large-latest",
    True
)
```

The mistralai python SDK provides methods to query the API for a list of [all available models](https://docs.mistral.ai/getting-started/models/models_overview/#api-versioning),
including paid endpoints for [premium models](https://docs.mistral.ai/getting-started/models/models_overview/) in addition to open weights.

## Text Generation

Once you've created your Outlines `Mistral` model instance, you're all set to generate text with this provider. You can simply call the model with a prompt.

For instance:

```python
import mistralai
import outlines

# Create the model
model = outlines.from_mistral(
    mistralai.Mistral(),
    "mistral-large-latest"
)

# Call it to generate text
response = model("What's the capital of Latvia?", max_tokens=20)
print(response) # 'Riga'
```

#### Vision

Some Mistral models support vision input. To use this feature, provide a list containing a text prompt and `Image` instances.

For instance:

```python
import io
import requests
import PIL
import outlines
import mistralai
from outlines.inputs import Image

# Create the model
model = outlines.from_mistral(
    mistralai.Mistral(),
    "mistral-large-latest"
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

You can also use chat inputs with the `Mistral` model. To do so, call the model with a `Chat` instance. The content of messsage within the chat can be vision inputs as described above.

For instance:

```python
import io
import requests
import PIL
import mistralai
import outlines
from outlines.inputs import Chat, Image

# Create the model
model = outlines.from_mistral(
    mistralai.Mistral(),
    "mistral-large-latest"
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

Finally, the `Mistral` model supports streaming through the `stream` method.

For instance:

```python
import mistralai
import outlines

# Create the model
model = outlines.from_mistral(
    mistralai.Mistral(),
    "mistral-large-latest"
)

# Stream the response
for chunk in model.stream("Tell me a short story about a cat.", max_tokens=50):
    print(chunk) # 'Once...'
```

## Structured Generation

Mistral provides supports for some forms of structured output: JSON schemas and JSON syntax. To use it, call the model with an `output_type` on top of your prompt.

#### JSON Schema

```python
from typing import List
from pydantic import BaseModel
import mistralai
import outlines

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_mistral(
    mistralai.Mistral(),
    "mistral-large-latest"
)

# Call it with the output type to generate structured text
result = model("Create a character, use the json format.", Character, top_p=0.1)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

#### JSON Syntax

What we mean by JSON syntax is what is sometimes called JSON mode, meaning that the model will return a valid JSON, but you do not get to specify its structure. To use this JSON mode, provide the `dict` type as an output type.

```python
import mistralai
import outlines

## Create the model
model = outlines.from_mistral(
    mistralai.Mistral(),
    "mistral-large-latest"
)


# Call it with the output type to generate structured text
result = model("Create a character, use the json format.", dict, temperature=0.5)
print(result) # '{"first_name": "Henri", "last_name": "Smith", "height": "170"}'
```

## Asynchronous Calls

All features presented above for the sync model are also available for the async model.

For instance:

```python
import asyncio
import mistralai
import outlines
from pydantic import BaseModel
from typing import List

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create the model
model = outlines.from_mistral(
    mistralai.Mistral(),
    "mistral-large-latest",
    True
)

async def text_generation():
    # Regular generation
    response = await model("What's the capital of Latvia?", max_tokens=20)
    print(response) # 'Riga'

    # Streaming
    async for chunk in  model.stream("Tell me a short story about a cat.", max_tokens=50):
        print(chunk, end="") # 'Once...'

    # Structured generation
    result = await model("Create a character, use the json format.", Character, top_p=0.1)
    print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
    print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']

asyncio.run(text_generation())
```

## Inference arguments

When calling the model, you can provide keyword arguments that will be passed down to the `chat.complete` method of the Mistral client and its async and streaming equivalents. Some of the most common arguments include `max_tokens`, `temperature`, `stop` and `top_p`.

Another keyword argument of interest is `n`. If set with an integer value superior to 1, Mistral will generate several sample responses and you will receive a list of strings as a response to your model call.

See the [Mistral API documentation](https://docs.mistral.ai/api/#tag/chat) for the full list of available arguments.


## Troubleshooting

- **ImportError: No module named 'mistralai'**
  → Run `pip install mistralai`.

- **Authentication Error**
  → Verify `MISTRAL_API_KEY` is set and valid. Test with the [Mistral Playground](https://chat.mistral.ai).

- **Schema Error (e.g., "Mistral does not support your schema")**
  → Ensure no `pattern` fields in Pydantic (Outlines sets `additionalProperties: false`); try a simpler schema or a different Outlines model (local models in particular).

- **Model Not Found Error**
  → Confirm the model name (e.g., `"mistral-small-latest"`) and your subscription tier. Check [docs](https://docs.mistral.ai/getting-started/models/).

- **Rate Limits or Quotas**
  → Monitor usage in the Mistral console; upgrade your plan for higher limits.

- **Input Validation Errors**
  → Ensure Chat messages use valid roles (`system`, `user`, `assistant`); list inputs start with strings.


*Last updated: October 2, 2025*
