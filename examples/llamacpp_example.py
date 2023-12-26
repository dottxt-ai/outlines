from enum import Enum

import torch
from pydantic import BaseModel, constr

import outlines


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
    name: constr(max_length=10)
    age: int
    armor: Armor
    weapon: Weapon
    strength: int


if __name__ == "__main__":
    # Download model from https://huggingface.co/TheBloke/phi-2-GGUF
    model = outlines.models.llamacpp("./phi-2.Q3_K_M.gguf", device="cpu")

    # Construct guided sequence generator
    generator = outlines.generate.json(model, Character, max_tokens=512)

    # Draw a sample
    rng = torch.Generator(device="cpu")
    rng.manual_seed(789005)

    prompt = "Instruct: You are a leading role play gamer. You have seen thousands of different characters and their attributes.\nPlease return a JSON object with common attributes of an RPG character. Give me a character description\nOutput:"

    sequence = generator(prompt, rng=rng)
    print(sequence)
