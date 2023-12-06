---
title: Outlines
hide:
  - navigation
  - toc
  - feedback
---

#

<figure markdown>
  ![Image title](assets/images/logo.png){ width="300" }
</figure>

<center>
    <h1 class="title">Generate text with LLMs</h1>
    <h2 class="subtitle">Robust prompting & (guided) text generation</h2>
    [:fontawesome-solid-bolt: Get started](get_started.md){ .md-button .md-button--primary }
    [:fontawesome-solid-code-pull-request: Contribute](https://github.com/outlines-dev/outlines){ .md-button }

<div class="index-pre-code">
```python
from enum import Enum
from pydantic import BaseModel, constr

import outlines.models as models
import outlines.text.generate as generate


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    strength: int


model = models.transformers("mistralai/Mistral-7B-v0.1", device="cuda")
generator = generate.json(model, Character, max_tokens=100)
sequence = generator("Give me a character description")
```
</div>

<a class="github-button" href="https://github.com/outlines-dev/outlines" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star outlines-dev/outlines on GitHub">Star</a>
<script async defer src="https://buttons.github.io/buttons.js"></script>
</center>
