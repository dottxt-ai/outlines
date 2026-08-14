---
title: Transformers MultiModal
---

# Transformers MultiModal

The Outlines `TransformersMultiModal` model inherits from `Transformers` and shares most of its interface. Please start by reading the [Transformers documentation](./transformers.md) as this document only focuses on the specificities of `TransformersMultiModal` compared to `Transformers`.

## Model Initialization

To load the model, you can use the `from_transformers` function. It takes 2 arguments:

- `model`: a `transformers` model (created with `AutoModelForImageTextToText` for instance)
- `tokenizer_or_processor`: a `transformers` processor (created with `AutoProcessor` for instance, it must be an instance of `ProcessorMixin`)
- `device_dtype` (optional): the tensor dtype to use for inference. If not provided, the model will use the default dtype.

For instance:

```python
import outlines
from transformers import AutoModelForImageTextToText, AutoProcessor

# Create the transformers model and processor
hf_model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
hf_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Create the Outlines model
model = outlines.from_transformers(hf_model, hf_processor)
```

## Model Input

As with other multimodal models, you should provide a list containing a text prompt and assets (`Image`, `Audio` or `Video` instances) as the model input. The type of asset to provide depends on the capabilities of the `transformers` model you are running.

Here's an example of using a vision multimodal model:

```python
from io import BytesIO
from urllib.request import urlopen

from PIL import Image as PILImage
from pydantic import BaseModel
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

import outlines
from outlines.inputs import Image

TEST_MODEL = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"

class Animal(BaseModel):
    specie: str
    color: str
    weight: int

def get_image_from_url(image_url):
    img_byte_stream = BytesIO(urlopen(image_url).read())
    image = PILImage.open(img_byte_stream).convert("RGB")
    image.format = "PNG"
    return image

# Create a model
model = outlines.from_transformers(
    LlavaForConditionalGeneration.from_pretrained(TEST_MODEL),
    AutoProcessor.from_pretrained(TEST_MODEL),
)

# Call it with a model input dict containing a text prompt and an image + an output type
result = model(
    ["<image>Describe this animal.", Image(get_image_from_url(IMAGE_URL))],
    Animal,
    max_new_tokens=100
)
print(result) # '{"specie": "cat", "color": "white", "weight": 4}'
print(Animal.model_validate_json(result)) # specie=cat, color=white, weight=4
```
!!! Warning

    Make sure your prompt contains the tags expected by your processor to correctly inject the assets in the prompt. For some vision multimodal models for instance, you need to add as many `<image>` tags in your prompt as there are image assets included in your model input. `Chat` method, instead, does not require this step.


### Chat
The `Chat` interface offers a more convenient way to work with multimodal inputs. You don't need to manually add asset tags like `<image>`. The model's HF processor handles the chat templating and asset placement for you automatically.
To do so, call the model with a `Chat` instance using a multimodal chat format. Assets must be pre-processed as `outlines.inputs.{Image, Audio, Video}` format, and only `image`, `video`, and `audio` types are supported.

For instance:

```python
import outlines
from outlines.inputs import Chat, Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image as PILImage
from io import BytesIO
from urllib.request import urlopen
import torch

model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    }

def get_image_from_url(image_url):
    img_byte_stream = BytesIO(urlopen(image_url).read())
    image = PILImage.open(img_byte_stream).convert("RGB")
    image.format = "PNG"
    return image

# Create the model
model = outlines.from_transformers(
    AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", **model_kwargs),
    AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", **model_kwargs)
)

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"

# Create the chat mutimodal input
prompt = Chat([
    {
        "role": "user",
        "content": [
            {"type": "image", "image": Image(get_image_from_url(IMAGE_URL))},
            {"type": "text", "text": "Describe the image in few words."}
        ],
    }
])

# Call the model to generate a response
response = model(prompt, max_new_tokens=50)
print(response) # 'A Siamese cat with blue eyes is sitting on a cat tree, looking alert and curious.'
```

Or using a list containing text and assets:

```python
import outlines
from outlines.inputs import Chat, Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image as PILImage
from io import BytesIO
import requests
import torch


TEST_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Function to get an image
def get_image(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    r = requests.get(url, headers=headers)
    image = PILImage.open(BytesIO(r.content)).convert("RGB")
    image.format = "PNG"
    return image

model_kwargs = {
        "torch_dtype": torch.bfloat16,
        # "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    }

# Create a model
model = outlines.from_transformers(
    AutoModelForImageTextToText.from_pretrained(TEST_MODEL, **model_kwargs),
    AutoProcessor.from_pretrained(TEST_MODEL, **model_kwargs),
)

# Create the chat input
prompt = Chat([
    {"role": "user", "content": "You are a helpful assistant that helps me described pictures."},
    {"role": "assistant", "content": "I'd be happy to help you describe pictures! Please go ahead and share an image"},
    {
        "role": "user",
        "content": ["Describe briefly the image", Image(get_image("https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"))]
    },
])

# Call the model to generate a response
response = model(prompt, max_new_tokens=50)
print(response) # 'The image shows a light-colored cat with a white chest...'
```


### Batching
The `TransformersMultiModal` model supports batching through the `batch` method. To use it, provide a list of prompts (using the formats described above) to the `batch` method. You will receive as a result a list of completions.

An example using the Chat format:

```python
import outlines
from outlines.inputs import Chat, Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image as PILImage
from io import BytesIO
from urllib.request import urlopen
import torch
from pydantic import BaseModel

model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    }

class Animal(BaseModel):
    animal: str
    color: str

def get_image_from_url(image_url):
    img_byte_stream = BytesIO(urlopen(image_url).read())
    image = PILImage.open(img_byte_stream).convert("RGB")
    image.format = "PNG"
    return image

# Create the model
model = outlines.from_transformers(
    AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", **model_kwargs),
    AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", **model_kwargs)
)

IMAGE_URL_1 = "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
IMAGE_URL_2 = "https://upload.wikimedia.org/wikipedia/commons/a/af/Golden_retriever_eating_pigs_foot.jpg"

# Create the chat mutimodal messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in few words."},
            {"type": "image", "image": Image(get_image_from_url(IMAGE_URL_1))},
        ],
    },
]

messages_2 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in few words."},
            {"type": "image", "image": Image(get_image_from_url(IMAGE_URL_2))},
        ],
    },
]

prompts = [Chat(messages), Chat(messages_2)]

# Call the model to generate a response
responses = model.batch(prompts, output_type=Animal, max_new_tokens=100)
print(responses) # ['{ "animal": "cat", "color": "white and gray" }', '{ "animal": "dog", "color": "white" }']
print([Animal.model_validate_json(i) for i in responses]) # [Animal(animal='cat', color='white and gray'), Animal(animal='dog', color='white')]
```


An example using a list of lists with tag assets:

```python
from io import BytesIO
from urllib.request import urlopen

from PIL import Image as PILImage
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

import outlines
from outlines.inputs import Image

TEST_MODEL = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
IMAGE_URL_2 ="https://upload.wikimedia.org/wikipedia/commons/9/98/Aldrin_Apollo_11_original.jpg"

def get_image_from_url(image_url):
    img_byte_stream = BytesIO(urlopen(image_url).read())
    image = PILImage.open(img_byte_stream).convert("RGB")
    image.format = "PNG"
    return image

# Create a model
model = outlines.from_transformers(
    LlavaForConditionalGeneration.from_pretrained(TEST_MODEL),
    AutoProcessor.from_pretrained(TEST_MODEL),
)

# Call the batch method with a list of model input dicts
result = model.batch(
    [
        ["<image>Describe the image.", Image(get_image_from_url(IMAGE_URL))],
        ["<image>Describe the image.", Image(get_image_from_url(IMAGE_URL_2))],
    ]
)
print(result) # ['The image shows a cat', 'The image shows an astronaut']
```
