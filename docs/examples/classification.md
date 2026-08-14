# Classification

Classification is a classic problem in NLP and finds many applications: spam detection, sentiment analysis, triaging of incoming requests, etc. We will use the example of a company that wants to sort support requests between those that require immediate attention (`URGENT`), those that can wait a little (`STANDARD`). You could easily extend the example by adding new labels.


This tutorial shows how one can implement multi-label classification using Outlines.

As always, we start with initializing the model. Since we are GPU poor we will be using a quantized version of Mistal-7B-v0.1:

```python
import outlines
import transformers

MODEL_NAME = "TheBloke/Mistral-7B-OpenOrca-AWQ"

model = outlines.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME),
    transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
)
```

We will use a prompt template stored in a text file:

```python
from outlines import Template

customer_support = Template.from_file("prompt_templates/classification.txt")
```

## Choosing between multiple choices

Outlines provides a convenient way to do multi-label classification, passing a Literal type hint to the `outlines.Generator` object:

```python
from typing import Literal
import outlines

generator = outlines.Generator(model, Literal["URGENT", "STANDARD"])

```
Outlines supports batched requests, so we will pass two requests to the model:

```python
requests = [
    "My hair is one fire! Please help me!!!",
    "Just wanted to say hi"
]

prompts = [customer_support(request=request) for request in requests]
```

We can now ask the model to classify the requests:

```python
labels = generator(prompts)
print(labels)
# ['URGENT', 'STANDARD']
```

## Using JSON-structured generation

Another (convoluted) way to do multi-label classification is to JSON-structured generation in Outlines. We first need to define our Pydantic schema that contains the labels:

```python
from enum import Enum
from pydantic import BaseModel


class Label(str, Enum):
    urgent = "URGENT"
    standard = "STANDARD"


class Classification(BaseModel):
    label: Label
```

We can then create a generator with the Pydantic model we just defined and call it:

```python
generator = outlines.Generator(model, Classification)
labels = generator(prompts)
print(labels)
# ['{"label":"URGENT"}', '{ "label": "STANDARD" }']
```
