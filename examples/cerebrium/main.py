from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines


model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2"),
    AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2"),
)


schema = {
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {"title": "Name", "maxLength": 10, "type": "string"},
        "age": {"title": "Age", "type": "integer"},
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {"title": "Strength", "type": "integer"},
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string",
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string",
        },
    },
}


def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    character = model(
        f"<s>[INST]Give me a character description. Describe {prompt}.[/INST]",
        outlines.json_schema(schema),
    )

    print(character)
    return character
