# Vision-Language Models with Outlines
This guide demonstrates how to use Outlines with vision-language models, leveraging the new transformers_vision module. Vision-language models can process both text and images, allowing for tasks like image captioning, visual question answering, and more.

We will be using the Pixtral-12B model from Mistral to take advantage of some of its visual reasoning capabilities and a workflow to generate a multistage atomic caption.

## Setup
First, we need to install the necessary dependencies. In addition to Outlines, we'll need to install the transformers library and any specific requirements for the vision-language model we'll be using.

```bash
pip install outlines transformers torch
```

### Initializing the Model
We'll use the transformers_vision function to initialize our vision-language model. This function is specifically designed to handle models that can process both text and image inputs. Today we'll be using the Pixtral model with the llama tokenizer. (Currently the mistral tokenizer is pending support).

```python
import torch
from transformers import (
    LlavaForConditionalGeneration,
)
model_name="mistral-community/pixtral-12b" # original magnet model is able to be loaded without issue
model_class=LlavaForConditionalGeneration

def get_vision_model(model_name: str, model_class: VisionModel):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    }
    processor_kwargs = {
        "device": "cuda",
    }

    model = outlines.models.transformers_vision(
        model.model_name,
        model_class=model.model_class,
        model_kwargs=model_kwargs,
        processor_kwargs=processor_kwargs,
    )
    return model
model = get_vision_model(model_name, model_class)
```

### Defining the Schema
Next, we'll define a schema for the output we expect from our vision-language model. This schema will help structure the model's responses.


```python
from pydantic import BaseModel, Field, confloat, constr
from pydantic.types import StringConstraints, PositiveFloat
from typing import List
from typing_extensions import Annotated

from enum import StrEnum
class TagType(StrEnum):
    ENTITY = "Entity"
    RELATIONSHIP = "Relationship"
    STYLE = "Style"
    ATTRIBUTE = "Attribute"
    COMPOSITION = "Composition"
    CONTEXTUAL = "Contextual"
    TECHNICAL = "Technical"
    SEMANTIC = "Semantic"

class ImageTag(BaseModel):
    tag: Annotated[
        constr(min_length=1, max_length=30),
        Field(
            description=(
                "Descriptive keyword or phrase representing the tag."
            )
        )
    ]
    category: TagType
    confidence: Annotated[
        confloat(le=1.0),
        Field(
            description=(
                "Confidence score for the tag, between 0 (exclusive) and 1 (inclusive)."
            )
        )
    ]

class ImageData(BaseModel):
    tags_list: List[ImageTag] = Field(..., min_items=8, max_items=20)
    short_caption: Annotated[str, StringConstraints(min_length=10, max_length=150)]
    dense_caption: Annotated[str, StringConstraints(min_length=100, max_length=2048)]

image_data_generator = outlines.generate.json(model, ImageData)
```

This schema defines the structure for image tags, including categories like Entity, Relationship, Style, etc., as well as short and dense captions.

### Preparing the Prompt

We'll create a prompt that instructs the model on how to analyze the image and generate the structured output:


```python
pixtral_instruction = """
<s>[INST]
<Task>You are a structured image analysis agent. Generate comprehensive tag list, caption, and dense caption for an image classification system.</Task>
<TagCategories requirement="You should generate a minimum of 1 tag for each category." confidence="Confidence score for the tag, between 0 (exclusive) and 1 (inclusive).">
- Entity : The content of the image, including the objects, people, and other elements.
- Relationship : The relationships between the entities in the image.
- Style : The style of the image, including the color, lighting, and other stylistic elements.
- Attribute : The most important attributes of the entities and relationships in the image.
- Composition : The composition of the image, including the arrangement of elements.
- Contextual : The contextual elements of the image, including the background, foreground, and other elements.
- Technical : The technical elements of the image, including the camera angle, lighting, and other technical details.
- Semantic : The semantic elements of the image, including the meaning of the image, the symbols, and other semantic details.
<Examples note="These show the expected format as an abstraction.">
{
  "tags_list": [
    {
      "tag": "subject 1",
      "category": "Entity",
      "confidence": 0.98
    },
    {
      "tag": "subject 2",
      "category": "Entity",
      "confidence": 0.95
    },
    {
      "tag": "subject 1 runs from subject 2",
      "category": "Relationship",
      "confidence": 0.90
    },
   }
</Examples>
</TagCategories>
<ShortCaption note="The short caption should be a concise single sentence caption of the image content with a maximum length of 100 characters.">
<DenseCaption note="The dense caption should be a descriptive but grounded narrative paragraph of the image content with high quality narrative prose. It should incorporate elements from each of the tag categories to provide a broad dense caption">\n[IMG][/INST]
""".strip()
```

This prompt provides detailed instructions to the model on how to generate comprehensive tag lists, captions, and dense captions for image analysis. Because of the ordering of the instructions the original tag generation serves as a sort of visual grounding for the captioning task, reducing the amount of manual post processing required.

### Generating Structured Output
Now we can use our model to generate structured output based on an input image:

```python
def img_from_url(url):
    img_byte_stream = BytesIO(urlopen(url).read())
    return Image.open(img_byte_stream).convert("RGB")

image_url="https://upload.wikimedia.org/wikipedia/commons/9/98/Aldrin_Apollo_11_original.jpg"
image= img_from_url(image_url)
result = image_data_generator(
    pixtral_instruction,
    [image]
)
print(result)
```

This code loads an image from a URL, passes it to our vision-language model along with the instruction prompt, and generates a structured output based on the defined schema. We end up with an output like this, ready to be used for the next stage in your pipeline:

```json
{'tags_list': [{'tag': 'astronaut',
   'category': <TagType.ENTITY: 'Entity'>,
   'confidence': 0.99},
  {'tag': 'moon', 'category': <TagType.ENTITY: 'Entity'>, 'confidence': 0.98},
  {'tag': 'space suit',
   'category': <TagType.ATTRIBUTE: 'Attribute'>,
   'confidence': 0.97},
  {'tag': 'lunar module',
   'category': <TagType.ENTITY: 'Entity'>,
   'confidence': 0.95},
  {'tag': 'shadow of astronaut',
   'category': <TagType.COMPOSITION: 'Composition'>,
   'confidence': 0.95},
  {'tag': 'footprints in moon dust',
   'category': <TagType.CONTEXTUAL: 'Contextual'>,
   'confidence': 0.93},
  {'tag': 'low angle shot',
   'category': <TagType.TECHNICAL: 'Technical'>,
   'confidence': 0.92},
  {'tag': 'human first steps on the moon',
   'category': <TagType.SEMANTIC: 'Semantic'>,
   'confidence': 0.95}],
 'short_caption': 'First man on the Moon',
 'dense_caption': "The figure clad in a pristine white space suit, emblazoned with the American flag, stands powerfully on the moon's desolate and rocky surface. The lunar module, a workhorse of space engineering, looms in the background, its metallic legs sinking slightly into the dust where footprints and tracks from the mission's journey are clearly visible. The photograph captures the astronaut from a low angle, emphasizing his imposing presence against the desolate lunar backdrop. The stark contrast between the blacks and whiteslicks of lost light and shadow adds dramatic depth to this seminal moment in human achievement."}
```

## Conclusion
The transformers_vision module in Outlines provides a powerful way to work with vision-language models. It allows for structured generation of outputs that combine image analysis with natural language processing, opening up possibilities for complex tasks like detailed image captioning, visual question answering, and more.

By leveraging the capabilities of models like Pixtral-12B and the structured output generation of Outlines, you can create sophisticated applications that understand and describe visual content in a highly structured and customizable manner.
