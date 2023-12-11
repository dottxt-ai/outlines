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
    [:fontawesome-brands-discord: Join the Community](https://discord.gg/ZxBxyWmW5n){ .md-button }

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


model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.json(model, Character, max_tokens=100)
sequence = generator("Give me a character description")
```
</div>
</center>
