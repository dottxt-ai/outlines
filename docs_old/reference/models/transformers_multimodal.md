# Transformers Multimodal

Outlines allows seamless use of [multimodal models](https://huggingface.co/learn/multimodal-models-course/en/unit1/introduction).

`outlines.models.TransformersMultimodal` shares interfaces with, and is based on [outlines.models.Transformers](./transformers.md).

## Create a `TransformersMultimodal` model

`TransformersMultimodal` models inherit from `Transformers` and accept similar initialization parameters, with the exception that you should provide a processor instead of a tokenizer.

You can use `outlines.from_transformers` to load a `transformers` model and processor:

```python
from transformers import CLIPModel, CLIPProcessor
from outlines import models

model_name = "openai/clip-vit-base-patch32"
model = models.from_transformers(
    CLIPModel.from_pretrained(model_name),
    CLIPProcessor.from_pretrained(model_name)
)
```

You must provider a processor adapted to your model. We currently offer official support for vision and audio models only, but you can use this class for any type of multimodal model (for instance video) as long as you provide the appropriate processor for your transformers model.

## Use the model to generate text from prompts and assets

When calling the model, the prompt argument you provide must be a dictionary using the following format:
```
{
    "text": Union[str, List[str]],
    "...": Any
}
```

The `text` key is mandatory. Other keys are optional and depend on the processor you are using.
The format and values of those keys must correspond to what you would provide to the `__call__` method of the processor if you were to call it directly. Thus, the text prompt must include the appropriate tags to indicate where the assets should be inserted (e.g. `<image>` or `<|AUDIO|>`).

Example of a correctly formatted prompt for a vision model:
```
{
    "text": "<image>Describe this image in one sentence:",
    "images": PIL.Image.Image
}
```

### Vision models

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
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from outlines import models

MODEL_NAME = "trl-internal-testing/tiny-LlavaForConditionalGeneration"

model = models.from_transformers(
    LlavaForConditionalGeneration.from_pretrained(MODEL_NAME),
    LlavaProcessor.from_pretrained(MODEL_NAME)
)
prompt = {
    "text": "<image> detailed description:",
    "images": img_from_url("https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg")
}
model(prompt)
```

You can include several images per prompt by adding more `<image>` tags to the prompt. Batching is also supported.
```python
from transformers import LlavaForConditionalGeneration
from outlines import models

MODEL_NAME = "trl-internal-testing/tiny-LlavaForConditionalGeneration"

model = models.from_transformers(
    LlavaForConditionalGeneration.from_pretrained(MODEL_NAME),
    LlavaProcessor.from_pretrained(MODEL_NAME)
)
prompt = {
    "text": ["<image><image>detailed description:", "<image><image>. What animals are present?"],
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

### Audio models

For easier use, we recommend you to create a convenience function to turn a `.wav` file from a URL into an audio array.
```python
import soundfile
from io import BytesIO
from urllib.request import urlopen

def audio_from_url(url):
    audio_data = BytesIO(urlopen(url).read())
    audio_array, _ = soundfile.read(audio_data)
    return audio_array
```

You can then call the model with your prompt and audio files to generate text.
```python
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from outlines import models

MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"

tf_model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_NAME)
tf_processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = models.TransformersMultiModal(tf_model, tf_processor)

text = """<|im_start|>user
Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
<|im_end|>
<|im_start|>user
Write a sentence about what you hear in the audio.<|im_end|>
<|im_start|>assistant
"""
prompt = {
    "text": text,
    "audios": audio_from_url(URL)
}
model(prompt)
```

You can include multiple audio files by providing a list for the `audios` argument and adding more of the appropriate tags to the prompt. Batching is also supported.

## Use the model for structured generation

You can use the model to generate structured data by providing a value for the parameter `output_type` (the second positional argument of the `generate` method, right after the prompt).

You can use most common Python types along with the Outlines DSL types `JsonSchema`, `Regex` and `CFG`.

For instance to do classification, you can use the `Regex` type:
```python
from outlines import models
from outlines.types import Regex
from transformers import LlavaForConditionalGeneration, LlavaProcessor

MODEL_NAME = "trl-internal-testing/tiny-LlavaForConditionalGeneration"

model = models.from_transformers(
    LlavaForConditionalGeneration.from_pretrained(MODEL_NAME),
    LlavaProcessor.from_pretrained(MODEL_NAME)
)
pattern = "Mercury|Venus|Earth|Mars|Saturn|Jupiter|Neptune|Uranus|Pluto"
prompt = {
    "text": "<image>detailed description:",
    "images": img_from_url("https://upload.wikimedia.org/wikipedia/commons/e/e3/Saturn_from_Cassini_Orbiter_%282004-10-06%29.jpg"),
}
model(prompt, Regex(pattern))
```

Another example could be to generated a structured description of an image using the `Json` type:
```python
from outlines import models
from pydantic import BaseModel
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from typing import List, Optional

MODEL_NAME = "trl-internal-testing/tiny-LlavaForConditionalGeneration"

class ImageData(BaseModel):
    caption: str
    tags_list: List[str]
    object_list: List[str]
    is_photo: bool

model = models.from_transformers(
    LlavaForConditionalGeneration.from_pretrained(MODEL_NAME),
    LlavaProcessor.from_pretrained(MODEL_NAME)
)
pattern = "Mercury|Venus|Earth|Mars|Saturn|Jupiter|Neptune|Uranus|Pluto"
prompt = {
    "text": "<image>detailed description:",
    "images": img_from_url("https://upload.wikimedia.org/wikipedia/commons/e/e3/Saturn_from_Cassini_Orbiter_%282004-10-06%29.jpg"),
}
model(prompt, Json(ImageData))
```

## Resources

### Choosing a model
- https://mmbench.opencompass.org.cn/leaderboard
- https://huggingface.co/spaces/WildVision/vision-arena
