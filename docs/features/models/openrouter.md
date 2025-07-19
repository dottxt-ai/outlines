# Openrouter

!!! Installation

    [OpenRouter](https://openrouter.ai/docs/api-reference/overview) uses the same API as OpenAI, so both services are [interoperable](./openai_compatible.md) using the `openai` library. Install all optional dependencies of the `OpenAI` model with: `pip install outlines[openai]`.

    You also need to have an Openrouter API key. This API key must either be set as an environment variable called `OPENAI_API_KEY` or be provided to the `openai.OpenAI` class when instantiating it.

## Model Initialization

To create a model instance, you can use the `from_openai` function. It takes 2 arguments:

- `client`: an `openai.OpenAI` instance
- `model_name`: the name of the model you want to use, defined as `provider/model`

For instance:

```python
import outlines
import openai

# Create the client
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="OPENAI_API_KEY",
)

# Create the model
model = outlines.from_openai(
    client,
    "x-ai/grok-4"
)
```

Leaving an empty string in the model name field will lead OpenRouter to use your default model defined in [settings](https://openrouter.ai/settings/preferences).

The [OpenRouter](https://openrouter.ai/models) website lists available models. Keep in mind that some models do not support `json_schema` response formats and may return a 400 error code as a result.

## Related Documentation

For specific implementations that use OpenAI-compatible APIs:

- [OpenAI](./openai.md): The original OpenAI API implementation
- [OpenAI compatible API](./openai_compatible.md): Details on how to use OpenAI-compatible APIs
