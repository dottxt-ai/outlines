---
title: BlockRun
---

# BlockRun

!!! Installation

    You need to install the `blockrun-llm` library to be able to use the BlockRun API in Outlines. Install all optional dependencies of the `BlockRun` model with: `pip install "outlines[blockrun]"`.

## Model Initialization

To create a BlockRun model instance, you can use the `from_blockrun` function. It takes 2 arguments:

- `client`: a `blockrun_llm.LLMClient` instance
- `model_name`: the name of the model you want to use in subsequent model calls (optional)

For instance:

```python
from blockrun_llm import LLMClient
import outlines

# Create the BlockRun client
client = LLMClient()

# Create the model
model = outlines.from_blockrun(
    client,
    "openai/gpt-4o"
)
```

Check the [BlockRun documentation](https://blockrun.ai/models) for an up-to-date list of available models.

## Text Generation

Once you've created your Outlines `BlockRun` model instance, you're all set to generate text with this provider. You can simply call the model with a text prompt.

For instance:

```python
from blockrun_llm import LLMClient
import outlines

# Create the model
model = outlines.from_blockrun(
    LLMClient(),
    "openai/gpt-4o"
)

# Call it to generate text
response = model("What's the capital of Latvia?")
print(response) # 'Riga'
```

#### Chat

You can also use chat inputs with the `BlockRun` model. To do so, call the model with a `Chat` instance. 

For instance:

```python
from blockrun_llm import LLMClient
from outlines import from_blockrun
from outlines.inputs import Chat

# Create the model
model = from_blockrun(
    LLMClient(),
    "openai/gpt-4o"
)

# Create the chat input
prompt = Chat([
    {"role": "user", "content": "You are a helpful assistant."},
    {"role": "assistant", "content": "I'd be happy to help!"},
    {
        "role": "user",
        "content": "Tell me a joke."
    },
])

# Call the model to generate a response
response = model(prompt)
print(response)
```

#### Streaming

Finally, the `BlockRun` model supports streaming through the `stream` method.

For instance:

```python
from blockrun_llm import LLMClient
import outlines

# Create the model
model = outlines.from_blockrun(
    LLMClient(),
    "openai/gpt-4o"
)

# Stream the response
for chunk in model.stream("Tell me a short story about a cat."):
    print(chunk)
```

## Inference arguments

When calling the model or streaming, you can provide keyword arguments that will be passed down to the BlockRun client. Make sure to include all the arguments you need to configure the client's behavior to your expected behavior. Some of the most common arguments include `max_tokens`, `temperature`, and `top_p`.

See the [BlockRun documentation](https://blockrun.ai/docs) for the full list of available arguments.
