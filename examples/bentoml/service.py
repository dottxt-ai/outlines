import typing as t

import bentoml
from import_model import BENTO_MODEL_TAG, MODEL_ID

DEFAULT_SCHEMA = """{
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


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class Outlines:
    bento_model_ref = bentoml.models.get(BENTO_MODEL_TAG)

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import outlines

        self.model = outlines.from_transformers(
            AutoTokenizer.from_pretrained(MODEL_ID),
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        )

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Give me a character description.",
        json_schema: t.Optional[str] = DEFAULT_SCHEMA,
    ) -> t.Dict[str, t.Any]:
        import outlines

        generator = outlines.Generator(self.model, outlines.json_schema(json_schema))
        character = generator(prompt)

        return character
