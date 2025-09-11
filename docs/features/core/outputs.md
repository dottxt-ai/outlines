---
title: Outputs
---

# Outputs

## Overview

Outlines uses two objcets to contain model response: `Ouptut` and `StreamingOutput`.

They both have two fields:

- `content`: the raw text reponse returned by the model
- `tool_calls`: a list of `ToolCallOutput` or `StreamingToolCallOutput` instances if the model decided to call a tool instead of giving a response directly. This field can only have a value if you provided a list of tools to the model in the first place.

To access the text response from the model, you would thus typically only do `reponse.output`. In the case of streaming, it would give you a chunk of the response.

## Chat

If you are using a `Chat` input to call the model, you can add the `Output` you received from the model to your `Chat` instance to add a new message that will be part of the conversation provided to the model the next time you can it.

For instance:

```python
import transformers
import outlines
from outlines.inputs import Chat, Image

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

model = outlines.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID),
    transformers.AutoTokenizer.from_pretrained(MODEL_ID),
)

# Initialize the chat with a system message.
chat_prompt = Chat([
    {"role": "system", "content": "You are a helpful assistant."},
])

# Add a user message to the chat.
chat_prompt.add_user_message("What's the capital of Latvia?")

# Call the model with the chat input.
response = model(chat_prompt)
print(response.content) # 'The capital of Latvia is Riga.'

# Add the output to the chat.
chat_prompt.add_output(response)

# Add another user message to the chat and call the model again.
chat_prompt.add_user_message("How many inhabitants does it have?")
response = model(chat_prompt)
print(response.content) # '600,000'
```

## Tool Calls

As described above, the output you receive from the model can contain a list of `ToolCallOutput` or `StreamingToolCallOutput` instances for the `tool_calls` field if the model decided to first call tools.

A `ToolCallOutput` or `StreamingToolCallOutput` contains three fields:
- `name`: the name of the tool to call
- `id`: the id of the tool call to make. If provided, it should typically be included in the `ToolMessage` containing the tool response you would add to the `Chat`
- `args`: the arguments to provide to the tool to call. This is a dictionnary for regular call and a string for streaming calls (as it could contain only a chunk of the whole args)

See the section on [tools](./tools.md) for an explanation on how to use the `ToolCallOutput` to make a tool call.
