---
title: Transformers MultiModal
---

# Transformers MultiModal

The Outlines `TransformersMultiModal` model inherits from `Transformers` and shares most of its interface. Please start by reading the [Transformers documentation](./transformers.md) as this document only focuses on the specificities of `TransformersMultiModal` compared to `Transformers`.

On top of the difference of arguments to provide to the `from_transformers` function as described below, the only other specificity of the `TransformersMultiModal` model is the format of the prompt to use when calling the model. For all other elements such as the output types and inference arguments, it works exactly the same way as `Transformers`.

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

Instead of a string or a list of strings (for batch generation), you should provide a dictionary or a list of dictionaries as a prompt when calling the `TransformersMultiModal` model. The dictionary should contain key-value pairs for all elements required by your processor. `text`, that contains the text prompt, is the only mandatory field. The format of that argument is:

```python
{
    "text": Union[str, List[str]]
    "<other_keys_depending_on_your_processor>": Union[Any, List[Any]]
}
```

Some common keys to include for processors are `images` for vision models or `audios` for audio models. The value for those keys would respectively be an image object and an audio file (or lists of those if there are several assets or if you are using batch generation).

Here's an example of using a vision multimodal model:

```python
from io import BytesIO
from urllib.request import urlopen

from PIL import Image
from pydantic import BaseModel
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

import outlines

TEST_MODEL = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"

class Animal(BaseModel):
    specie: str
    color: str
    weight: int

def get_image_from_url():
    img_byte_stream = BytesIO(urlopen(IMAGE_URL).read())
    return Image.open(img_byte_stream).convert("RGB")

# Create a model
model = outlines.from_transformers(
    LlavaForConditionalGeneration.from_pretrained(TEST_MODEL),
    AutoProcessor.from_pretrained(TEST_MODEL),
)

# Call it with a model input dict containing a text prompt and an image + an output type
result = model(
    {"text": "<image>Describe this animal.", "images": get_image_from_url(IMAGE_URL)},
    Animal,
)
print(result) # '{"specie": "cat", "color": "white", "weight": 4}'
print(Animal.model_validate_json(result)) # specie=cat, color=white, weight=4
```

!!! Warning

    Make sure your prompt (the value of the `text` key) contains the tags expected by your processor to correctly inject the assets in the prompt. For some vision multimodal models for instance, you need to add as many `<image>` tags in your prompt as there are images present in the value of the `images` key.
