# Transformers Vision

Outlines allows seamless use of [vision models](https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/tasks-models-part1).

`outlines.models.Transformers_vision` shares interfaces with, and is based on [outlines.models.Transformers](./transformers.md).

## Create a `TransformersVision` model

`TransformersVision` models inherit from `Transformers` and accept the same initialization parameters.

In addition, they also accept the optional parameters `processor_class` and `processor_kwargs`. Those are used to create a `Processor` instance that is then used to preprocess the images. By default, `AutoProcessor` is used to create the processor as such: `AutoProcessor.from_pretrained(model_name, **processor_kwargs)`.

If your model is not compatible with `AutoProcessor`, you must provide a value for the `processor_class` parameter.
For instance:
```python
from outlines import models
from transformers import CLIPModel, CLIPProcessor

model = models.TransformersVision("openai/clip-vit-base-patch32", model_class=CLIPModel, processor_class=CLIPProcessor)
```

## Use the model to generate text from prompts and images

When calling the model, the prompt argument you provide must be a dictionary with a key `"prompts"` and a key `"images"`. The associated values must be a string or a list of strings for the prompts, and a PIL image or a list of PIL images for the images. Your prompts must include `<image>` tags tokens to indicate where the image should be inserted. There must be as many `<image>` tags as there are images.

For easier use, we recommend you to create a convenience function to load a `PIL.Image` from URL.
```python
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

def img_from_url(url):
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")
```

You can then call the model with your prompts and images to generate text.
```python
from transformers import LlavaForConditionalGeneration
from outlines import models

model = models.TransformersVision("trl-internal-testing/tiny-LlavaForConditionalGeneration", model_class=LlavaForConditionalGeneration)
prompt = {
    "prompts": "<image> detailed description:",
    "images": img_from_url("https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")
}
model(prompt)
```

You can include several images per prompt by adding more `<image>` tags to the prompt. Batching is also supported.
```python
from transformers import LlavaForConditionalGeneration
from outlines import models

model = models.TransformersVision("trl-internal-testing/tiny-LlavaForConditionalGeneration", model_class=LlavaForConditionalGeneration)
prompt = {
    "prompts": ["<image><image>detailed description:", "<image><image>. What animals are present?"],
    "images": [
        img_from_url("https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"),
        img_from_url("https://upload.wikimedia.org/wikipedia/commons/7/71/2010-kodiak-bear-1.jpg"),
        img_from_url("https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"),
        img_from_url("https://upload.wikimedia.org/wikipedia/commons/7/71/2010-kodiak-bear-1.jpg"),
    ]
}
model(prompt)
```

Here we have two prompts, each expecting two images. We correspondingly provide four images. This will generate two descriptions, one for each prompt.

### Use the model for structured generation

You can use the model to generate structured data by providing a value for the parameter `output_type` (the second positional argument of the `generate` method, right after the prompt).

Supported types include `Json`, `Choice`, `Regex` and `CFG`.

For instance to do classification, you can use the `Regex` type:
```python
from outlines import models
from outlines.types import Regex
from transformers import LlavaForConditionalGeneration

model = models.TransformersVision("trl-internal-testing/tiny-LlavaForConditionalGeneration", model_class=LlavaForConditionalGeneration)
pattern = "Mercury|Venus|Earth|Mars|Saturn|Jupiter|Neptune|Uranus|Pluto"
prompt = {
    "prompts": "<image>detailed description:",
    "images": img_from_url("https://upload.wikimedia.org/wikipedia/commons/e/e3/Saturn_from_Cassini_Orbiter_%282004-10-06%29.jpg"),
}
model(prompt, Regex(pattern))
```

Another example could be to generated a structured description of an image using the `Json` type:
```python
from outlines import models
from pydantic import BaseModel
from transformers import LlavaForConditionalGeneration
from typing import List, Optional

class ImageData(BaseModel):
    caption: str
    tags_list: List[str]
    object_list: List[str]
    is_photo: bool

model = models.TransformersVision("trl-internal-testing/tiny-LlavaForConditionalGeneration", model_class=LlavaForConditionalGeneration)
pattern = "Mercury|Venus|Earth|Mars|Saturn|Jupiter|Neptune|Uranus|Pluto"
prompt = {
    "prompts": "<image>detailed description:",
    "images": img_from_url("https://upload.wikimedia.org/wikipedia/commons/e/e3/Saturn_from_Cassini_Orbiter_%282004-10-06%29.jpg"),
}
model(prompt, Json(ImageData))
```

## Resources

### Choosing a model
- https://mmbench.opencompass.org.cn/leaderboard
- https://huggingface.co/spaces/WildVision/vision-arena
