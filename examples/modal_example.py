import modal

app = modal.App(name="outlines-app")


outlines_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "outlines==1.0.0",
    "transformers==4.38.2",
    "datasets==2.18.0",
    "accelerate==0.27.2",
)


def import_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    _ = AutoTokenizer.from_pretrained(model_id)
    _ = AutoModelForCausalLM.from_pretrained(model_id)


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


@app.function(image=outlines_image, gpu="A100-40GB")
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    model = outlines.from_transformers(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        model=AutoModelForCausalLM.from_pretrained(model_id, device="cuda"),
    )

    character = model(
        f"<s>[INST]Give me a character description. Describe {prompt}.[/INST]",
        outlines.json_schema(schema),
    )

    print(character)


@app.local_entrypoint()
def main(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    generate.remote(prompt)
