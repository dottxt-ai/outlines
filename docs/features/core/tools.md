---
title: Tools
---

# Tools

## Overview

Some models support tool calling, meaning that instead of directly providing its final response, the model can require to call tools you have defined and would later use the tool response in its final response. Tool calling typically goes along providing a `Chat` input as it implies a multiturn conversation with the model.

For the moment, tool calling is supported by three Outlines models:

- `Anthropic`
- `Gemini`
- `OpenAI`

## Tool Definition

Using tool calling starts with defining the tools that the model can call. There are three formats currently supported as described below.

Once defined, the tools must be provided in a list to the `tools` keyword argument to the `Generator` constructor or to the text generation methods of a model. As such, the interface for `tools` is very similar to that of the `output_type`.

#### ToolDef

A tool can first by defined as a dictionnary. A `ToolDef` dict must contain the following keys:

- `name`: The name of the tool
- `description`: A description of the tool to help the LLM understand its use
- `parameters`: A dictionnary containing the paramters of the tool, using the JSON properties format. If the LLM decides to call the tool, it will provide values for the parameters
- `required`: A list of parameters that are mandatory. All those parameters must be included in the `parameters` key described above

For instance:

```python
import openai
from outlines import from_openai
from outlines.inputs import Chat
from outlines.tools import ToolDef

client = openai.OpenAI()
model = from_openai(client, "gpt-4o")

chat = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Tokyo?"},
])

weather_tool = ToolDef(
    name="get_weather",
    description="Give the weather for a given city, and optionally for a specific hour of the day",
    parameters={"city": {"type": "string"}, "hour": {"type": "integer"}},
    required=["city"],
)

response = model(chat, tools=[weather_tool])
print(response.tool_calls) # [ToolCallOutput(name='get_weather', id='call_p7ToNwgrgoEk9poN7PXTELT5', args={'city': 'Tokyo'})]
```

#### Function

A python function can be used as a tool definition. The `description` would then correspond to the docstring while the `parameters` and `required` would be deduced from the signature.

```python
import openai
from outlines import from_openai
from outlines.inputs import Chat
from typing import Optional

client = openai.OpenAI()
model = from_openai(client, "gpt-4o")

chat = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Tokyo?"},
])

def get_weather(city: str, hour: Optional[int] = None):
    """Give the weather for a given city, and optionally for a specific hour of the day"""
    pass

response = model(chat, tools=[get_weather])
print(response.tool_calls) # [ToolCallOutput(name='get_weather', id='call_IdsfmBss6XhiBDbchTqp3HHz', args={'city': 'Tokyo'})]
```

#### Pydantic model

Lastly, you can use a Pydantic model to define the interface of your tool.

```python
import openai
from outlines import from_openai
from outlines.inputs import Chat
from pydantic import BaseModel
from typing import Optional

client = openai.OpenAI()
model = from_openai(client, "gpt-4o")

chat = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Tokyo?"},
])

class GetWeather(BaseModel):
    """Give the weather for a given city, and optionally for a specific hour of the day"""
    city: str
    hour: Optional[int] = None

response = model(chat, tools=[GetWeather])
print(response.tool_calls) # [ToolCallOutput(name='GetWeather', id='call_KWfADMEr6dnDDcw1m2dllRvq', args={'city': 'Tokyo'})]
```

## Tool Calls and Responses

If the model decides to call a tool, you'll get a value for the `tool_calls` attribute of the `Output` received. This value is a `OutputToolCall` instance containing three attributes:

- `name`: The name of the tool to call
- `id`: The id of the tool call to be able to easily link the tool call and the tool response
- `args`: A dictionnary containing for each parameter required by the tool the value provided by the LLM

You should use the `name` and the `args` to call your tool yourself and get its reponse. Afterward, you can add to your chat the `Output` you first receive and a `ToolMessage` before being able to call the model again to continue the conversation.

For instance:

```python
import openai
from outlines import Generator, from_openai
from outlines.inputs import Chat
from typing import Optional

# Our tool
def get_weather(city: str, hour: Optional[int] = None):
    """Give the weather for a given city, and optionally for a specific hour of the day"""
    return "20 degrees"

client = openai.OpenAI()
model = from_openai(client, "gpt-4o")
generator = Generator(model, tools=[get_weather])

chat = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Tokyo?"},
])

response = generator(chat)
print(response.tool_calls) # [ToolCallOutput(name='get_weather', id='call_NlIGHr8HoiVgSZfOJ7Y5xz35', args={'city': 'Tokyo'})]

# Add the model response to the chat
chat.add_output(response)

# Call the tool with the parameters given by the model and add a tool message to the chat
tool_call = response.tool_calls[0]
tool_response = get_weather(**tool_call.args)
chat.add_tool_message(
    content=tool_response,
    tool_name=tool_call.name,
    tool_call_id=tool_call.id
)

response = generator(chat)
print(response.content) # The weather in Tokyo is currently 20 degrees.
```

When using streaming, the response would be a `StreamingOutput` and the `tool_calls` value a list of `StreamingOutputToolCall`. The only difference compared to what's the describe above is that the `args` field would be a string as the value is received by chunks. You need to concatenate the chunks together to get the full `args` to use to call the tool.
