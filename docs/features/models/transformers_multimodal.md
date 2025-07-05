---
title: Transformers MultiModal
---

# Transformers MultiModal

The Outlines `TransformersMultiModal` model inherits from `Transformers` and shares most of its interface. Please start by reading the [Transformers documentation](./transformers.md) as this document only focuses on the specificities of `TransformersMultiModal` compared to `Transformers`.

## Model Initialization

To load the model, you can use the `from_transformers` function. It takes 2 arguments:

- `model`: a `transformers` model (created with `AutoModelForCausalLM` for instance)
- `tokenizer_or_processor`: a `transformers` processor (created with `AutoProcessor` for instance, it must be an instance of `ProcessorMixin`)

For instance:

```python
import outlines
from transformers import AutoModelForCausalLM, AutoProcessor

# Create the transformers model and processor
hf_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
hf_processor = AutoProcessor.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

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

The `TransformersMultiModal` model supports batch generation. To use it, invoke the `batch` method with a list of lists. You will receive as a result a list of completions.

For instance:

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

!!! Warning

    Make sure your prompt contains the tags expected by your processor to correctly inject the assets in the prompt. For some vision multimodal models for instance, you need to add as many `<image>` tags in your prompt as there are image assets included in your model input.
