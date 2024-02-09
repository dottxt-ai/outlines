# Named entity extraction

Named Entity Extraction is a fundamental problem in NLP. It involves identifying and categorizing named entities within a document: people, organization, dates, places, etc. It is usually the first step in a more complex NLP worklow. Here we will use the example of a pizza restaurant that receives orders via their website and need to identify the number and types of pizzas that are being ordered.

Getting LLMs to output the extracted entities in a structured format can be challenging. In this tutorial we will see how we can use Outlines' JSON-structured generation to extract entities from a document and return them in a valid JSON data structure 100% of the time.

As always, we start with initializing the model. We will be using a quantized version of Mistal-7B-v0.1 (we're GPU poor):

```python
import outlines

model = outlines.models.transformers("TheBloke/Mistral-7B-OpenOrca-AWQ", device="cuda")
```

And we will be using the following prompt template:

```python
@outlines.prompt
def take_order(order):
    """You are the owner of a pizza parlor. Customers \
    send you orders from which you need to extract:

    1. The pizza that is ordered
    2. The number of pizzas

    # EXAMPLE

    ORDER: I would like one Margherita pizza
    RESULT: {"pizza": "Margherita", "number": 1}

    # OUTPUT INSTRUCTIONS

    Answer in valid JSON. Here are the different objects relevant for the output:

    Order:
        pizza (str): name of the pizza
        number (int): number of pizzas

    Return a valid JSON of type "Order"

    # OUTPUT

    ORDER: {{ order }}
    RESULT: """
```

We now define our data model using Pydantic:

```python
from enum import Enum
from pydantic import BaseModel

class Pizza(str, Enum):
    margherita = "Margherita"
    pepperonni = "Pepperoni"
    calzone = "Calzone"

class Order(BaseModel):
    pizza: Pizza
    number: int
```

We can now define our generator and call it on several incoming orders:

```python
orders = [
    "Hi! I would like to order two pepperonni pizzas and would like them in 30mins.",
    "Is it possible to get 12 margheritas?"
]
prompts = [take_order(order) for order in orders]

generator = outlines.generate.json(model, Order)

results = generator(prompts)
print(results)
# [Order(pizza=<Pizza.pepperonni: 'Pepperoni'>, number=2),
#  Order(pizza=<Pizza.margherita: 'Margherita'>, number=12)]
```

There are several ways you could improve this example:

- Clients may order several types of pizzas.
- Clients may order drinks as well.
- If the pizza place has a delivery service we need to extract the client's address and phone number
- Clients may specify the time for which they want the pizza. We could then check against a queuing system and reply to them with the estimated delivery time.

How would you change the Pydantic model to account for these use cases?
