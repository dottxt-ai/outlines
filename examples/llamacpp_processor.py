from enum import Enum

from llama_cpp import Llama, LogitsProcessorList
from pydantic import BaseModel, constr

from outlines.generate.processors import JSONLogitsProcessor
from outlines.models.llamacpp import LlamaCppTokenizer


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
    llama = Llama("./phi-2.Q4_K_M.gguf")
    tokenizer = LlamaCppTokenizer(llama)

    prompt = "Instruct: You are a leading role play gamer. You have seen thousands of different characters and their attributes.\nPlease return a JSON object with common attributes of an RPG character. Give me a character description\nOutput:"

    logits_processor = JSONLogitsProcessor(Character, tokenizer)

    json_str = llama.create_completion(
        prompt,
        top_k=40,
        top_p=0.95,
        temperature=0.7,
        max_tokens=100,
        logits_processor=LogitsProcessorList([logits_processor]),
    )["choices"][0]["text"]

    print(json_str)
