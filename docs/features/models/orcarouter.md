# OrcaRouter

!!! Installation

    [OrcaRouter](https://www.orcarouter.ai) uses the same API as OpenAI, so both services are [interoperable](./openai_compatible.md) using the `openai` library. Install all optional dependencies of the `OpenAI` model with: `pip install "outlines[openai]"`.

    You also need to have an OrcaRouter API key. This API key must either be set as an environment variable called `OPENAI_API_KEY` or be provided to the `openai.OpenAI` class when instantiating it.

## Model Initialization

To create a model instance, you can use the `from_openai` function. It takes 2 arguments:

- `client`: an `openai.OpenAI` instance
- `model_name`: the name of the model you want to use, defined as `provider/model`

For instance:

```python
import os
import outlines
import openai

# Create the client
client = openai.OpenAI(
    base_url="https://api.orcarouter.ai/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Create the model
model = outlines.from_openai(
    client,
    "openai/gpt-5.5"
)
```

You can also pass `"orcarouter/auto"` as the model name to let OrcaRouter pick an upstream for each request according to the routing strategy you configured in the [routing console](https://www.orcarouter.ai/console/routing).

The [OrcaRouter](https://www.orcarouter.ai/models) website lists available models. Keep in mind that some models do not support `json_schema` response formats and may return a 400 error code as a result.

## Related Documentation

For specific implementations that use OpenAI-compatible APIs:

- [OpenAI](./openai.md): The original OpenAI API implementation
- [OpenAI compatible API](./openai_compatible.md): Details on how to use OpenAI-compatible APIs
