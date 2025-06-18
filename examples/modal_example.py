import modal

app = modal.App(name="outlines-app")


outlines_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.0.37",
    "transformers==4.38.2",
    "datasets==2.18.0",
    "accelerate==0.27.2",
)


def import_model():
    import outlines

    outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")


outlines_image = outlines_image.run_function(import_model)


schema = """{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "maxLength": 10,
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}"""


@app.function(image=outlines_image, gpu=modal.gpu.A100(memory=80))
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    import outlines

    model = outlines.models.transformers("mistralai/Mistral-7B-v0.1", device="cuda")

    generator = outlines.generate.json(model, schema)
    character = generator(
        f"<s>[INST]Give me a character description. Describe {prompt}.[/INST]"
    )

    print(character)


@app.local_entrypoint()
def main(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    generate.remote(prompt)
