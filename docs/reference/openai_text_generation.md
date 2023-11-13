# Generate text with the OpenAI API

Outlines is focused on 🔓 models, but includes an OpenAI integration nevertheless. You can instantiate a model very simply by calling the [outlines.models.openai][] function, with either a chat or non chat model:

```python
from outlines import models

model = models.openai("text-davinci-003")
model = models.openai("gpt4")

print(type(model))
# OpenAIAPI
```

!!! note

    It is currently not possible to pass a system message to the model. If that is something you need, please [open an Issue](https://github.com/outlines-dev/outlines/issues) or, better, [submit a Pull Request](https://github.com/outlines-dev/outlines/pulls).

The OpenAI integration supports the following features:

- The ability to stop the generation when a specified sequence is found [🔗](#stop-when-a-sequence-is-found)
- The ability to choose between different choices [🔗](#multiple-choices)
- Vectorization, i.e. the ability to pass an array of prompts and execute all requests concurrently [🔗](#vectorized-calls)

## Stop when a sequence is found

The OpenAI API tends to be chatty and it can be useful to stop the generation once a given sequence has been found, instead of paying for the extra tokens and needing to post-process the output. For instance if you only to generate a single sentence:

```python
from outlines import models

model = models.openai("text-davinci-003")
response = model("Write a sentence", stop_at=['.'])
```

## Multiple choices

It can be difficult to deal with a classification problem with the OpenAI API. However well you prompt the model, chances are you are going to have to post-process the output anyway. Sometimes the model will even make up choices. Outlines allows you to *guarantee* that the output of the model will be within a set of choices you specify:

```python
from outlines import models

prompt = """
Review: The OpenAI API is very limited. It does not allow me to do guided generation properly.
Question: What is the overall sentiment of this review?
Answer:
"""

model = models.openai("text-davinci-003")
response = model(prompt, is_in=['Positive', 'Negative'])
```

## Vectorized calls

A unique feature of Outlines is that calls to the OpenAI API are *vectorized* (In the [NumPy sense](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html) of the word). In plain English this means that you can call an Openai model with an array of prompts with arbitrary shape to an OpenAI model and it will return an array of answers. All calls are executed concurrently, which means this takes roughly the same time as calling the model with a single prompt:

```python
from outlines import models
from outlines import text

@text.prompt
def template(input_numbers):
    """Use these numbers and basic arithmetic to get 24 as a result:

    Input: {{ input_numbers }}
    Steps: """

prompts = [
    template([1, 2, 3]),
    template([5, 9, 7]),
    template([10, 12])
]

model = models.openai("text-davinci-003")
results = model(prompts)
print(results.shape)
# (3,)

print(type(results))
# <class 'numpy.ndarray'>

print(results)
# [
#     "\n1. 1 + 2 x 3 = 7\n2. 7 + 3 x 4 = 19\n3. 19 + 5 = 24",
#     "\n1. Add the three numbers together: 5 + 9 + 7 = 21\n2. Subtract 21 from 24: 24 - 21 = 3\n3. Multiply the remaining number by itself: 3 x 3 = 9\n4. Add the number with the multiplication result: 21 + 9 = 24",
#    "\n\n1. Add the two numbers together: 10 + 12 = 22 \n2. Subtract one of the numbers: 22 - 10 = 12 \n3. Multiply the two numbers together: 12 x 12 = 144 \n4. Divide the first number by the result: 144 / 10 = 14.4 \n5. Add the initial two numbers together again: 14.4 + 12 = 26.4 \n6. Subtract 2: 26.4 - 2 = 24",
# ]
```

Beware that in this case the output of the model is a NumPy array. So if you want to concatenate the prompt to the result you have to use `numpy.char.add`:

```python
import numpy as np

new_prompts = np.char.add(prompts, results)
print(new_prompts)

# [
#     "Use these numbers and basic arithmetic to get 24 as a result:\n\nInput: [1, 2, 3]\nSteps:\n1. 1 + 2 x 3 = 7\n2. 7 + 3 x 4 = 19\n3. 19 + 5 = 24",
#     "Use these numbers and basic arithmetic to get 24 as a result:\n\nInput: [5, 9, 7]\nSteps:\n1. Add the three numbers together: 5 + 9 + 7 = 21\n2. Subtract 21 from 24: 24 - 21 = 3\n3. Multiply the remaining number by itself: 3 x 3 = 9\n4. Add the number with the multiplication result: 21 + 9 = 24",
#    "'Use these numbers and basic arithmetic to get 24 as a result:\n\nInput: [10, 12]\nSteps:\n\n1. Add the two numbers together: 10 + 12 = 22 \n2. Subtract one of the numbers: 22 - 10 = 12 \n3. Multiply the two numbers together: 12 x 12 = 144 \n4. Divide the first number by the result: 144 / 10 = 14.4 \n5. Add the initial two numbers together again: 14.4 + 12 = 26.4 \n6. Subtract 2: 26.4 - 2 = 24",
# ]
```

You can also ask for several samples for a single prompt:

```python
from outlines import models
from outlines import text


@text.prompt
def template(input_numbers):
    """Use these numbers and basic arithmetic to get 24 as a result:

    Input: {{ input_numbers }}
    Steps:"""


model = models.openai("text-davinci-003")
results = model(template([1, 2, 3]), samples=3, stop_at=["\n2"])
print(results.shape)
# (3,)

print(results)
# [
#     ' \n1. Subtract 1 from 3',
#     '\n1. Add the three numbers: 1 + 2 + 3 = 6',
#     ' (1 + 3) x (2 + 2) = 24'
# ]
```

Or ask for several samples for an array of prompts. In this case *the last dimension is the sample dimension*:

```python
from outlines import models
from outlines import text


@text.prompt
def template(input_numbers):
    """Use these numbers and basic arithmetic to get 24 as a result:

    Input: {{ input_numbers }}
    Steps:"""


prompts = [template([1, 2, 3]), template([5, 9, 7]), template([10, 12])]

model = models.openai("text-davinci-003")
results = model(prompts, samples=2, stop_at=["\n2"])
print(results.shape)
# (3, 2)

print(results)
# [
#     ['\n1. Add the numbers: 1 + 2 + 3 = 6', ' (3 * 2) - 1 = 5\n        5 * 4 = 20\n        20 + 4 = 24'],
#     ['\n\n1. (5 + 9) x 7 =  56', '\n1. 5 x 9 = 45'],
#     [' \n1. Add the two numbers together: 10 + 12 = 22', '\n1. Add 10 + 12']
# ]
```

You may find this useful, e.g., to implement [Tree of Thoughts](https://arxiv.org/abs/2305.10601).

!!! note

    Outlines provides an `@outlines.vectorize` decorator that you can use on any `async` python function. This can be useful for instance when you call a remote API within your workflow.
