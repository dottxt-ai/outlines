# Classification

Classification is a classic problem in NLP and finds many applications: spam detection, sentiment analysis, triaging of incoming requests, etc. We will use the example of a company that wants to sort support requests between those that require immediate attention (`URGENT`), those that can wait a little (`STANDARD`). You could easily extend the example by adding new labels.


This tutorial shows how one can implement multi-label classification using Outlines. We will use two functionalities of the library: `generate.choice` and `generate.json`.

As always, we start with initializing the model. Since we are GPU poor we will be using a quantized version of Mistal-7B-v0.1:

```python
import outlines

model = outlines.models.transformers("TheBloke/Mistral-7B-OpenOrca-AWQ", device="cuda")
```

We will use the following prompt template:

```python
@outlines.prompt
def customer_support(request):
    """You are an experienced customer success manager.

    Given a request from a client, you need to determine when the
    request is urgent using the label "URGENT" or when it can wait
    a little with the label "STANDARD".

    # Examples

    Request: "How are you?"
    Label: STANDARD

    Request: "I need this fixed immediately!"
    Label: URGENT

    # TASK

    Request: {{ request }}
    Label: """
```

## Choosing between multiple choices

Outlines provides a shortcut to do multi-label classification, using the `outlines.generate.choice` function to initialize a generator. Outlines uses multinomial sampling by default, here we will use the greedy sampler to get the label with the highest probability:

```python
from outlines.generate.samplers import greedy

generator = outlines.generate.choice(model, ["URGENT", "STANDARD"], sampler=greedy)
```
Outlines supports batched requests, so we will pass two requests to the model:

```python
requests = [
    "My hair is one fire! Please help me!!!",
    "Just wanted to say hi"
]

prompts = [customer_support(request) for request in requests]
```

We can now asks the model to classify the requests:

```python
labels = generator(prompts)
print(labels)
# ['URGENT', 'STANDARD']
```

Now, you might be in a hurry and don't want to wait until the model finishes completion. After all, you only need to see the first letter of the response to know whether the request is urgent or standard. You can instead stream the response:

```python
tokens = generator.stream(prompts)
labels = ["URGENT" if "U" in token else "STANDARD" for token in next(tokens)]
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

and we can use `generate.json` by passing this Pydantic model we just defined, and call the generator:

```python
generator = outlines.generate.json(model, Classification, sampler=greedy)
labels = generator(prompts)
print(labels)
# [Classification(label=<Label.urgent: 'URGENT'>), Classification(label=<Label.standard: 'STANDARD'>)]
```
