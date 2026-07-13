# Requesty

!!! Installation

    [Requesty](https://docs.requesty.ai) uses the same API as OpenAI, so both services are [interoperable](./openai_compatible.md) using the `openai` library. Install all optional dependencies of the `OpenAI` model with: `pip install "outlines[openai]"`.

    You also need to have a Requesty API key. You can create one at [app.requesty.ai/api-keys](https://app.requesty.ai/api-keys). This API key must either be set as an environment variable called `OPENAI_API_KEY` or be provided to the `openai.OpenAI` class when instantiating it.

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
    base_url="https://router.requesty.ai/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Create the model
model = outlines.from_openai(
    client,
    "openai/gpt-4o-mini"
)
```

The [Requesty](https://app.requesty.ai/router/list) website lists available models. Keep in mind that some models do not support `json_schema` response formats and may return a 400 error code as a result.

## Related Documentation

For specific implementations that use OpenAI-compatible APIs:

- [OpenAI](./openai.md): The original OpenAI API implementation
- [OpenAI compatible API](./openai_compatible.md): Details on how to use OpenAI-compatible APIs
