# Generate text with the OpenAI API

!!! Installation

    You need to install the `openai` and `tiktoken` libraries to be able to use the OpenAI API in Outlines.

Outlines supports models available via the OpenAI Chat API, e.g. ChatGPT and GPT-4. The following models can be used with Outlines:

```python
from outlines import models

model = models.openai("gpt-3.5-turbo")
model = models.openai("gpt-4")

print(type(model))
# OpenAI
```


It is possible to pass a system message to the model when initializing it:

```python
from outlines import models

model = models.openai("gpt-4", system_prompt="You are a useful assistant")
```

This message will be used for every subsequent use of the model:

## Monitoring API use

It is important to be able to track your API usage when working with OpenAI's API. The number of prompt tokens and completion tokens is directly accessible via the model instance:

```python
import outlines.models

model = models.openai("gpt-4")

print(model.prompt_tokens)
# 0

print(model.completion_tokens)
# 0
```

These numbers are updated every time you call the model.


## Advanced usage

It is possible to specify the values for `seed`, `presence_penalty`, `frequence_penalty`, `top_p` by passing an instance of `OpenAIConfig` when initializing the model:

```python
from outlines.models.openai import OpenAIConfig
from outlines import models

config = OpenAIConfig(
    presence_penalty=1.,
    frequence_penalty=1.,
    top_p=.95,
    seed=0,
)
model = models.openai("gpt-4", config=config)
```
