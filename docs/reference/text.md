# Text generation

Outlines provides a unified interface to generate text with many language models, API-based and local:

```python
from outlines import models, generate

model = models.openai("gpt-4")
generator = generate.text(model)
answer = generator("What is 2+2?")

model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.text(model)
answer = generator("What is 2+2?")
```

We generate text in two steps:

1. Instantiate a generator with the model you want to use
2. Call the generator with the prompt


## Limit the number of tokens generated

To limit the number of tokens generated you can pass the `max_tokens` positional argument to the generator:

```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.text(model)

answer = generator("What is 2+2?", 5)
answer = generator("What is 2+2?", max_tokens=5)
```

## Stop when a given string is found

You can also ask the model to stop generating text after a given string has been generated, for instance a period or a line break. You can pass a string or a line of string for the `stop_at` argument:


```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.text(model)

answer = generator("What is 2+2?", stop_at=".")
answer = generator("What is 2+2?", stop_at=[".", "\n"])
```

## Streaming

Outlines allows you to stream the model's response by calling the `.stream` method of the generator with the prompt:


```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.text(model)

tokens = generator.stream("What is 2+2?")
for token in tokens:
    print(token)
```

## Use a different sampler

Outlines uses the multinomial sampler by default. To specify another sampler, for instance the greedy sampler you need to specify it when instantiating the generator:

```python
from outlines import models, generate
from outlines.generate.samplers import greedy


model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.text(model, sampler=greedy)

tokens = generator("What is 2+2?")
```
