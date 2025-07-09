---
title: SGLang
---

# SGLang

## Prerequisites

The Outlines `SGLang` model is intended to be used along with an SGLang instance running on a separate server (can be local or remote). Make sure you have a SGLang server running and accessible before using the `SGLang` model. For instance by running:

```shell
pip install "sglang[all]"

python -m sglang.launch_server \
  --model-path NousResearch/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

Follow the [Installation instructions](https://docs.sglang.ai/start/install.html) for more information on how to set up a SGLang server for your particular setup.

As the SGLang client relies on the `openai` python sdk, you need to have the `openai` package installed. Install all optional dependencies of the `SGLang` model with: `pip install outlines[sglang]`.

When launching your SGLang server, you can specify the backend engine to use for structured generation through the `grammar-backend` cli argument. Add `--grammar-backend outlines` to your command to use Outlines instead of the default engine.

## Model Initialization

To load the model, you can use the `from_sglang` function. The argument of the function is either an `OpenAI` or `AsyncOpenAI` instance from the `openai` library. Make sure the value of the `base_url` argument of the `OpenAI` client points to your running SGLang server. Consult the [SGLang documentation](https://docs.sglang.ai/backend/send_request.html) on using an OpenAI client with an SGLang server for more information.

Based on whether the `openai` client instance is synchronous or asynchronous, you will receive a `SGLang` or `AsyncSGLang` model instance.

For instance:

```python
import openai
import outlines

# Create the OpenAI client
sync_openai_client = openai.OpenAI(base_url="http://localhost:11434")
async_openai_client = openai.AsyncOpenAI(base_url="http://localhost:11434")

# Create a sync model
sync_model = outlines.from_sglang(sync_openai_client)
print(type(sync_model)) # <class 'outlines.models.sglang.SGLang'>

# Create an async model
async_model = outlines.from_sglang(async_openai_client)
print(type(async_model)) # <class 'outlines.models.sglang.AsyncSGLang'>
```

## Text Generation

To generate text, you can simply call the model with a prompt.

For instance:

```python
import openai
import outlines

# Create the model
model = outlines.from_openai(openai.OpenAI(base_url="http://localhost:11434"))

# Call it to generate text
response = model("What's the capital of Latvia?", max_tokens=20)
print(response) # 'Riga'
```

#### Vision

Some models you can run with SGLang support vision input. To use this feature, provide a list containing a text prompt and `Image` instances.

For instance:

```python
import io
import requests
import PIL
import outlines
import openai
from outlines.inputs import Image

# Create the model
model = outlines.from_openai(openai.OpenAI(base_url="http://localhost:11434"))

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

You can also use chat inputs with the `SGLang` model. To do so, call the model with a `Chat` instance. The content of messsage within the chat can be vision inputs as described above.

For instance:

```python
import io
import requests
import PIL
import openai
import outlines
from outlines.inputs import Chat, Image

# Create the model
model = outlines.from_openai(openai.OpenAI(base_url="http://localhost:11434"))

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

Finally, the `SGLang` model supports streaming through the `stream` method.

For instance:

```python
import openai
import outlines

# Create the model
model = outlines.from_openai(openai.OpenAI(base_url="http://localhost:11434"))

# Stream the response
for chunk in model.stream("Tell me a short story about a cat.", max_tokens=50):
    print(chunk) # 'Once...'
```

## Structured Generation

SGLang supports all output types available in Outlines (context-free grammars with caveats though, see the subsection below for more details). Simply provide an `output_type` after the prompt when calling the model. All structured generation features work with both synchronous and asynchronous models.

### Simple Type

```python
import openai
import outlines

output_type = int

openai_client = openai.OpenAI(base_url="http://localhost:11434")
model = outlines.from_sglang(openai_client)

result = model("How many countries are there in the world?", output_type)
print(result) # '200'
```

### JSON Schema

```python
import openai
import outlines
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

openai_client = openai.OpenAI(base_url="http://localhost:11434")
model = outlines.from_sglang(openai_client)

result = model("Create a character.", Character, frequency_penalty=1.5)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
```

### Multiple Choice

```python
from typing import Literal
import openai
import outlines

output_type = Literal["Paris", "London", "Rome", "Berlin"]

openai_client = openai.OpenAI(base_url="http://localhost:11434")
model = outlines.from_sglang(openai_client)

result = model("What is the capital of France?", output_type, temperature=0)
print(result) # 'Paris'
```

### Regex

```python
import openai
import outlines
from outlines.types import Regex

output_type = Regex(r"\d{3}-\d{2}-\d{4}")

openai_client = openai.OpenAI(base_url="http://localhost:11434")
model = outlines.from_sglang(openai_client)

result = model("Generate a fake social security number.", output_type, top_p=0.1)
print(result) # '782-32-3789'
```

### Context-Free Grammar

SGLang supports grammars, but expects an EBNF format instead of the Lark format Outlines uses. Thus, to use a context-free grammar with SGLang, provide a string using the EBNF syntax to the Outlines `CFG` object.

```python
import openai
import outlines
from outlines.types import CFG

ebnf_grammar = """
root ::= answer
answer ::= "yes" | "no"
"""
output_type = CFG(ebnf_grammar)

openai_client = openai.OpenAI(base_url="http://localhost:11434")
model = outlines.from_sglang(openai_client)

result = model("Is the weather good today?", output_type)
print(result) # 'yes'
```

### Async Structured Generation

All structured generation features work seamlessly with async models:

```python
import asyncio
import openai
import outlines
from typing import List
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int

async def generate_user():
    async_client = openai.AsyncOpenAI(base_url="http://localhost:11434")
    async_model = outlines.from_sglang(async_client)

    result = await async_model("Generate a random user profile.", output_type=User)
    user = User.model_validate_json(result)
    print(f"Name: {user.name}, Email: {user.email}, Age: {user.age}")

asyncio.run(generate_user())
```

## Inference Arguments

When calling the model, you can provide optional parameters on top of the prompt and the output type. Those will be passed on to the `chat.completions.create` method of the OpenAI client.

An optional parameter of particular interest is `extra_body`, which is a dictionary containing arguments that are specific to SGLang and are not part of the standard `openai` interface.

See the [SGLang documentation](https://docs.sglang.ai/backend/openai_api_completions.html) on parameters for the OpenAI-compatible server for more information on inference parameters.
