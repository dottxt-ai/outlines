import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

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

generator = outlines.generate.json(model, schema)


def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    character = generator(
        f"<s>[INST]Give me a character description. Describe {prompt}.[/INST]"
    )

    print(character)
    return character
