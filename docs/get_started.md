---
title: Get Started
---

<!-- Logo -->
<figure markdown>
  ![Outlines logo](assets/images/logo.png){ width="300" }
</figure>

<!-- Badges -->
<div align="center">

<p><i>Generate text that machines understand.</i></p>

<a href="https://pypi.org/project/outlines/"><img src="https://img.shields.io/pypi/v/outlines?color=ECEFF4&logo=python&logoColor=white&style=flat-square"></img></a>
<a href="https://github.com/outlines-dev/outlines/graphs/contributors"><img src="https://img.shields.io/github/contributors/outlines-dev/outlines?style=flat-square&logo=github&logoColor=white&color=ECEFF4"></img></a>
<a href="https://x.com/dottxtai"><img src="https://img.shields.io/twitter/follow/dottxtai?style=social"></img></a>
</div>

<!-- Remove the title -->
#

## :sparkles: Features

- :material-keyboard: Prompting utilities
- :material-regex: Regex-guided generation
- :material-code-json: JSON-guided generation
- :material-dice-multiple-outline: Multiple sequence sampling methods
- :material-open-source-initiative: Integration with several open source libraries

## :floppy_disk: Install

```bash
pip install outlines
```

??? info "Using OpenAI and Transformers"

    Outlines :wavy_dash: does not install the `openai` or `transformers` libraries by default. You will have to install these libraries manually.

## :eyes: Sneak Peek

=== "Code"

    ```python
    from enum import Enum
    from pydantic import BaseModel, constr

    import outlines.models as models
    import outlines.text.generate as generate


    class Weapon(str, Enum):
        sword = "sword"
        axe = "axe"
        mace = "mace"
        spear = "spear"
        bow = "bow"
        crossbow = "crossbow"


    class Armor(str, Enum):
        leather = "leather"
        chainmail = "chainmail"
        plate = "plate"


    class Character(BaseModel):
        name: constr(max_length=20)
        age: int
        armor: Armor
        weapon: Weapon
        strength: int


    model = models.transformers("gpt2")
    generator = generate.json(model, Character)
    sequence = generator("Create a character description for a role playing game in JSON")

    print(sequence)
    ```
=== "Output"

    ```json
    {
      "name": "Anonymous Tokens",
      "age": 7,
      "armor": "plate",
      "weapon": "mace",
      "strength": 4171
    }
    `````

## Acknowledgements

<figure markdown>
  <a href="https://www.normalcomputing.ai">
  ![Normal Computing logo](assets/images/normal_computing.jpg){ width="150" }
  </a>
</figure>

Outlines was originally developed at [@NormalComputing](https://twitter.com/NormalComputing) by [@remilouf](https://twitter.com/remilouf) and [@BrandonTWillard](https://twitter.com/BrandonTWillard).
