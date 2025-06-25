# PDF to structured output with vision language models

A common task with language models is to ask language models questions about a PDF file.

Typically, the output is unstructured text, i.e. "talking" to your PDF.

In some cases, you may wish to extract structured information from the PDF, like tables, lists, citations, etc.

PDFs are difficult to machine read. However, you can simply convert the PDF to images, and then use a vision language model to extract structured information from the images.

This cookbook demonstrates how to

1. Convert a PDF to a list of images
2. Use a vision language model to extract structured information from the images

## Dependencies

You'll need to install these dependencies:

```shell
pip install outlines pillow transformers torch==2.4.0 pdf2image

# Optional, but makes the output look nicer
pip install rich
```

## Import the necessary libraries

```python
from PIL import Image
import outlines
import torch
from transformers import AutoProcessor
from pydantic import BaseModel
from typing import List, Optional
from pdf2image import convert_from_path
import os
from rich import print
import requests
```

## Choose a model

We've tested this example with [Pixtral 12b](https://huggingface.co/mistral-community/pixtral-12b) and [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

To use Pixtral:

```python
from transformers import LlavaForConditionalGeneration, LlavaProcessor
model_name="mistral-community/pixtral-12b"
model_class=LlavaForConditionalGeneration
processor_class = LlavaProcessor
```

To use Qwen-2-VL:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
model_name = "Qwen/Qwen2-VL-7B-Instruct"
model_class = Qwen2VLForConditionalGeneration
processor_class = AutoProcessor
```

You can load your model into memory with:

```python
# This loads the model into memory. On your first run,
# it will have to download the model, which might take a while.
model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16}
processor_kwargs={"device_map": "cpu"}
tf_model = model_class.from_pretrained(model_name, **model_kwargs)
tf_processor = processor_class.from_pretrained(model_name, **processor_kwargs)

model = outlines.from_transformers(tf_model, tf_processor)
```

## Convert the PDF to images

We'll use the `pdf2image` library to convert each page of the PDF to an image.

`convert_pdf_to_images` is a convenience function that converts each page of the PDF to an image, and optionally saves the images to disk when `output_dir` is provided.

Note: the `dpi` argument is important. It controls the resolution of the images. High DPI images are higher quality and may yield better results,
but they are also larger, slower to process, and require more memory.

```python
from pdf2image import convert_from_path
from PIL import Image
import os
from typing import List, Optional

def convert_pdf_to_images(
    pdf_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 120,
    fmt: str = 'PNG'
) -> List[Image.Image]:
    """
    Convert a PDF file to a list of PIL Image objects.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional directory to save the images
        dpi: Resolution for the conversion. High DPI is high quality, but also slow and memory intensive.
        fmt: Output format (PNG recommended for quality)

    Returns:
        List of PIL Image objects
    """
    # Convert PDF to list of images
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        fmt=fmt
    )

    # Optionally save images
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, image in enumerate(images):
            image.save(os.path.join(output_dir, f'page_{i+1}.{fmt.lower()}'))

    return images
```

We're going to use the [Louf & Willard paper](https://arxiv.org/pdf/2307.09702) that described the method that Outlines uses for structured generation.

To download the PDF, run:

```python
# Download the PDF file
pdf_url = "https://arxiv.org/pdf/2307.09702"
response = requests.get(pdf_url)

# Save the PDF locally
with open("louf-willard.pdf", "wb") as f:
    f.write(response.content)
```

Now, we can convert the PDF to a list of images:

```python
# Load the pdf
images = convert_pdf_to_images(
    "louf-willard.pdf",
    dpi=120,
    output_dir="output_images"
)
```

## Extract structured information from the images

The structured output you can extract is exactly the same as everywhere else in Outlines -- you can use regular expressions, JSON schemas, selecting from a list of options, etc.

### Extracting data into JSON

Suppose you wished to go through each page of the PDF, and extract the page description, key takeaways, and page number.

You can do this by defining a JSON schema, and then using `outlines.Generator` to extract the data.

First, define the structure you want to extract:

```python
class PageSummary(BaseModel):
    description: str
    key_takeaways: List[str]
    page_number: int
```

Second, we need to set up the prompt. Adding special tokens can be tricky, so we use the transformers processor to apply the special tokens for us. To do so, we specify a list of messages, where each message is a dictionary with a `role` and `content` key.

Images are denoted with `type: "image"`, and text is denoted with `type: "text"`.

```python
messages = [
    {
        "role": "user",
        "content": [
            # The text you're passing to the model --
            # this is where you do your standard prompting.
            {"type": "text", "text": f"""
                Describe the page in a way that is easy for a PhD student to understand.

                Return the information in the following JSON schema:
                {PageSummary.model_json_schema()}

                Here is the page:
                """
            },

            # This a placeholder, the actual image is passed in when
            # we call the generator function down below.
            {"type": "image", "image": ""},
        ],
    }
]

# Convert the messages to the final prompt
prompt = tf_processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

Now we iterate through each image, and extract the structured information:

```python
# Page summarizer function
page_summary_generator = outlines.Generator(model, PageSummary)

for image in images:
    result = page_summary_generator({"text": prompt, "images": image})
    print(result)
```

### Regular expressions to extract the arxiv paper identifier

The [arXiv paper identifier](https://info.arxiv.org/help/arxiv_identifier.html) is a unique identifier for each paper. These identifiers have the format `arXiv:YYMM.NNNNN` (five end digits) or `arXiv:YYMM.NNNN` (four end digits). arXiv identifiers are typically watermarked on papers uploaded to arXiv.

arXiv identifiers are optionally followed by a version number, i.e. `arXiv:YYMM.NNNNNvX`.

We can use a regular expression to define this patter:

```python
from outlines.types import Regex

paper_regex = Regex(r'arXiv:\d{2}[01]\d\.\d{4,5}(v\d)?')
```

We can build an extractor function from the regex:

```python
id_extractor = outlines.Generator(model, paper_regex)
```

Now, we can extract the arxiv paper identifier from the first image:

```python
arxiv_instruction = tf_processor.apply_chat_template(
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
                Extract the arxiv paper identifier from the page.

                Here is the page:
                """},
                {"type": "image", "image": ""},
            ],
        }
    ],
    tokenize=False,
    add_generation_prompt=True
)

# Extract the arxiv paper identifier
paper_id = id_extractor({"text": arxiv_instruction, "images": images[0]})
```

As of the time of this writing, the arxiv paper identifier is

```
arXiv:2307.09702v4
```

Your version number may be different, but the part before `vX` should match.

### Categorize the paper into one of several categories

`outlines.Generator` also allows the model to select one of several options by providing a Literal type hint with the categories.

Suppose we wanted to categorize the paper into being about "language models", "cell biology", or "other". We would then define the output type as `Literal["llms", "cell biology", "other"]`.

Let's define a few categories we might be interested in:

```python
categories = [
    "llms",
    "cell biology",
    "other"
]
```

Now we can construct the prompt:

```python
categorization_instruction = tf_processor.apply_chat_template(
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
                Please choose one of the following categories
                that best describes the paper.

                {categories}

                Here is the paper:
                """},

                {"type": "image", "image": ""},
            ],
        }
    ],
    tokenize=False,
    add_generation_prompt=True
)
```

Now we can show the model the first page and extract the category:

```python
from typing import Literal

# Build the choice extractor
categorizer = outlines.Generator(model, Literal["llms", "cell biology", "other"])

# Categorize the paper
category = categorizer({"text": categorization_instruction, "images": images[0]})
print(category)
```

Which should return:

```
llms
```

## Additional notes

You can provide multiple images to the model by

1. Adding additional image messages
2. Providing a list of images to the generator

For example, to have two images, you can do:

```python
two_image_prompt = tf_processor.apply_chat_template(
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "are both of these images of hot dogs?"},

                # Tell the model there are two images
                {"type": "image", "image": ""},
                {"type": "image", "image": ""},
            ],
        }
    ],
    tokenize=False,
    add_generation_prompt=True
)

# Pass two images to the model
generator = outlines.Generator(model, Literal["hot dog", "not hot dog"])

result = generator({"text": two_image_prompt, "images": [images[0], images[1]]})
print(result)
```

Using the first to pages of the paper (they are not images of hot dogs), we should get

```
not hot dog
```
