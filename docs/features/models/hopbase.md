# HopBase

!!! Installation

    [HopBase](https://hop-base.com) uses the same API as OpenAI, so both services are [interoperable](./openai_compatible.md) using the `openai` library. Install all optional dependencies of the `OpenAI` model with: `pip install "outlines[openai]"`.

    You also need to have a HopBase API key. You can create one in the [HopBase console](https://api.hop-base.com/login). This API key must either be set as an environment variable called `OPENAI_API_KEY` or be provided to the `openai.OpenAI` class when instantiating it.

## Model Initialization

To create a model instance, you can use the `from_openai` function. It takes 2 arguments:

- `client`: an `openai.OpenAI` instance
- `model_name`: the name of the model you want to use, for instance `gpt-6.1-sol` or `glm-5.3`

For instance:

```python
import os
import outlines
import openai

# Create the client
client = openai.OpenAI(
    base_url="https://api.hop-base.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Create the model
model = outlines.from_openai(
    client,
    "gpt-6.1-sol"
)
```

HopBase serves GPT, Claude and Gemini models alongside model families such as GLM, DeepSeek, Qwen and Kimi behind the same base URL, so the same Outlines code can target any of them by changing the model name. `client.models.list()` returns the model names your API key can call. The [HopBase models page](https://hop-base.com/models) lists the full catalog.

## Structured Generation

Structured generation works as described in the [OpenAI](./openai.md#structured-generation) documentation: JSON schemas (for instance a Pydantic model) and JSON syntax (`dict`) are supported. To constrain the output to a set of choices, use a `Literal` field inside a Pydantic model.

```python
from typing import Literal
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    skills: list[str]

class Sentiment(BaseModel):
    label: Literal["positive", "negative"]

result = model("Create a fantasy character.", Character, max_tokens=500)
print(Character.model_validate_json(result)) # name='Mira Thornvale' age=27 skills=['Herbalism', ...]

result = model("Is 'I love this' positive or negative?", Sentiment, max_tokens=500)
print(Sentiment.model_validate_json(result)) # label='positive'
```

Support for `json_schema` response formats depends on the model. Some models return a 400 error code, and some return valid JSON that does not follow the schema, so validate the output as shown above. For instance, `gpt-6.1-sol` follows the schema, while `glm-5.3` is better used with the `dict` output type. Reasoning models such as `glm-5.3` also need a generous `max_tokens`, otherwise the reasoning can use the whole budget and the answer comes back empty.

## Related Documentation

For specific implementations that use OpenAI-compatible APIs:

- [OpenAI](./openai.md): The original OpenAI API implementation
- [OpenAI compatible API](./openai_compatible.md): Details on how to use OpenAI-compatible APIs
