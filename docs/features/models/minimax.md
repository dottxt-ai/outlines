# MiniMax

!!! Installation

    MiniMax exposes an OpenAI-compatible API, so Outlines uses the `openai` library to communicate with it. Install all optional dependencies of the `OpenAI` model with: `pip install "outlines[openai]"`.

    You also need a MiniMax API key. This key must either be set as an environment variable called `OPENAI_API_KEY` or be provided to the `openai.OpenAI` class when instantiating it.

## Model Initialization

To create a MiniMax model instance, use the `from_minimax` function. It takes 2 arguments:

- `client`: an `openai.OpenAI` or `openai.AsyncOpenAI` instance whose `base_url` is set to a MiniMax endpoint
- `model_name`: the name of the model you want to use (optional; defaults to `MiniMax-M3.1-Flash`)

MiniMax provides two OpenAI-compatible regional endpoints, exposed as `MINIMAX_BASE_URLS`:

- `global_en`: `https://api.minimax.io/v1`
- `cn_zh`: `https://api.minimaxi.com/v1`

For instance:

```python
import outlines
import openai
from outlines.models.minimax import MINIMAX_BASE_URLS

# Create the client pointing at the region that matches your account
client = openai.OpenAI(
    api_key="MINIMAX_API_KEY",
    base_url=MINIMAX_BASE_URLS["global_en"],
)

# Create the model
model = outlines.from_minimax(
    client,
    "MiniMax-M3.1-Flash"
)
```

Use `MiniMax-M3.1-Flash` (the default) or `MiniMax-M3`. Flash has a 1,000,000-token context window and supports up to 524,288 output tokens. Explicit model names, including saved legacy and custom IDs, are passed through unchanged. A `model=` argument on a generation call overrides the constructor's model.

Flash uses adaptive thinking by default. You can pass `reasoning_effort="low"`, `"medium"`, `"high"`, `"xhigh"`, or `"max"` (the API default); `"none"` and disabled thinking are not supported for Flash. M3 accepts adaptive or disabled thinking and ignores reasoning effort. The adapter forwards explicit inference arguments without rewriting them. See the [MiniMax model documentation](https://platform.minimax.io/docs/guides/models-intro) for the current model contract.

## Text Generation

Once you've created your Outlines `MiniMax` model instance, you can generate text by calling the model with a prompt.

```python
import outlines
import openai
from outlines.models.minimax import MINIMAX_BASE_URLS

model = outlines.from_minimax(
    openai.OpenAI(base_url=MINIMAX_BASE_URLS["global_en"]),
)

result = model("What's the capital of Latvia?", max_tokens=20)
print(result) # 'Riga'
```

#### Vision

`MiniMax-M3` supports image input. To use this feature, provide a list containing a text prompt and `Image` instances.

```python
import io
import requests
import PIL
import outlines
import openai
from outlines.inputs import Image
from outlines.models.minimax import MINIMAX_BASE_URLS

model = outlines.from_minimax(
    openai.OpenAI(base_url=MINIMAX_BASE_URLS["global_en"]),
    "MiniMax-M3",
)

def get_image(url):
    r = requests.get(url)
    return PIL.Image.open(io.BytesIO(r.content))

prompt = [
    "Describe the image",
    Image(get_image("https://picsum.photos/id/237/400/300"))
]

response = model(prompt, max_tokens=50)
print(response)
```

#### Video

`MiniMax-M3` also supports video input. Provide a list containing a text prompt and `Video` instances holding the URL of the video to analyze.

```python
import outlines
import openai
from outlines.inputs import Video
from outlines.models.minimax import MINIMAX_BASE_URLS

model = outlines.from_minimax(
    openai.OpenAI(base_url=MINIMAX_BASE_URLS["global_en"]),
    "MiniMax-M3",
)

prompt = [
    "Describe the video",
    Video("https://example.com/clip.mp4"),
]

response = model(prompt, max_tokens=50)
print(response)
```

## Structured Generation

MiniMax uses the same OpenAI-compatible `response_format` payload as the `OpenAI` model, so JSON schema and JSON mode structured outputs are supported. Regex and grammar-based output types are not available.

```python
import outlines
import openai
from pydantic import BaseModel
from outlines.models.minimax import MINIMAX_BASE_URLS

class Character(BaseModel):
    name: str
    age: int

model = outlines.from_minimax(
    openai.OpenAI(base_url=MINIMAX_BASE_URLS["global_en"]),
    "MiniMax-M3",
)

result = model("Create a character.", Character)
print(result) # '{"name": "Aurora", "age": 34}'
```

## Related Documentation

- [OpenAI](./openai.md): the OpenAI model that MiniMax builds upon
- [OpenAI compatible API](./openai_compatible.md): details on OpenAI-compatible APIs
