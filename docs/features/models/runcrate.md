# Runcrate

!!! Installation

    [Runcrate](https://www.runcrate.ai/?utm_source=outlines&utm_medium=directory&utm_campaign=gpu_pricing) serves open models on an API that is the same as OpenAI's, so both services are [interoperable](./openai_compatible.md) using the `openai` library. Install all optional dependencies of the `OpenAI` model with: `pip install "outlines[openai]"`.

    You also need a Runcrate API key. You can create one in the [Runcrate dashboard](https://www.runcrate.ai/dashboard/api-keys?utm_source=outlines&utm_medium=directory&utm_campaign=gpu_pricing); open models are free with your GitHub account. This API key must either be set as an environment variable called `OPENAI_API_KEY` or be provided to the `openai.OpenAI` class when instantiating it.

## Model Initialization

To create a model instance, you can use the `from_openai` function. It takes 2 arguments:

- `client`: an `openai.OpenAI` instance
- `model_name`: the name of the model you want to use, defined as `vendor/model`

For instance:

```python
import os
import outlines
import openai

# Create the client
client = openai.OpenAI(
    base_url="https://api.runcrate.ai/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Create the model
model = outlines.from_openai(
    client,
    "deepseek-ai/DeepSeek-V4.1-Flash"
)
```

The public model list at `https://api.runcrate.ai/v1/models` gives every model id, for example `zai-org/GLM-5.3`, `Qwen/Qwen3.8-Flash` or `openai/gpt-oss-120b`. Keep in mind that some models do not support `json_schema` response formats and may return a 400 error code as a result.

## Related Documentation

For specific implementations that use OpenAI-compatible APIs:

- [OpenAI](./openai.md): The original OpenAI API implementation
- [OpenAI compatible API](./openai_compatible.md): Details on how to use OpenAI-compatible APIs
