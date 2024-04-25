import typing as t

import bentoml
from import_model import BENTO_MODEL_TAG

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

        import outlines

        self.model = outlines.models.transformers(
            self.bento_model_ref.path,
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Give me a character description.",
        json_schema: t.Optional[str] = DEFAULT_SCHEMA,
    ) -> t.Dict[str, t.Any]:
        import outlines

        generator = outlines.generate.json(self.model, json_schema)
        character = generator(prompt)

        return character
