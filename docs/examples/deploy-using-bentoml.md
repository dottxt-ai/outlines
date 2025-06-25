# Run Outlines using BentoML

[BentoML](https://github.com/bentoml/BentoML) is an open-source model serving library for building performant and scalable AI applications with Python. It comes with tools that you need for serving optimization, model packaging, and production deployment.

In this guide, we will show you how to use BentoML to run programs written with Outlines on GPU locally and in [BentoCloud](https://www.bentoml.com/), an AI Inference Platform for enterprise AI teams. The example source code in this guide is also available in the [examples/bentoml/](https://github.com/dottxt-ai/outlines/blob/main/examples/bentoml/) directory.

## Import a model

First we need to download an LLM (Mistral-7B-v0.1 in this example and you can use any other LLM) and import the model into BentoML's [Model Store](https://docs.bentoml.com/en/latest/guides/model-store.html). Let's install BentoML and other dependencies from PyPi (preferably in a virtual environment):

```shell
pip install -r requirements.txt
```

Then save the code snippet below as `import_model.py` and run `python import_model.py`.

**Note**: You need to accept related conditions on [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1) first to gain access to Mistral-7B-v0.1.

```python
import bentoml

MODEL_ID = "mistralai/Mistral-7B-v0.1"
BENTO_MODEL_TAG = MODEL_ID.lower().replace("/", "--")

def import_model(model_id, bento_model_tag):

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    with bentoml.models.create(bento_model_tag) as bento_model_ref:
        tokenizer.save_pretrained(bento_model_ref.path)
        model.save_pretrained(bento_model_ref.path)


if __name__ == "__main__":
    import_model(MODEL_ID, BENTO_MODEL_TAG)
```

You can verify the download is successful by running:

```shell
$ bentoml models list

Tag                                          Module  Size        Creation Time
mistralai--mistral-7b-v0.1:m7lmf5ac2cmubnnz          13.49 GiB   2024-04-25 06:52:39
```

## Define a BentoML Service

As the model is ready, we can define a [BentoML Service](https://docs.bentoml.com/en/latest/guides/services.html) to wrap the capabilities of the model.

We will run the JSON-structured generation example [in the README](https://github.com/dottxt-ai/outlines?tab=readme-ov-file#efficient-json-generation-following-a-json-schema), with the following schema:

```python
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
```

First, we need to define a BentoML service by decorating an ordinary class (`Outlines` here) with `@bentoml.service` decorator. We pass to this decorator some configuration and GPU on which we want this service to run in BentoCloud (here an L4 with 24GB memory):

```python
import typing as t
import bentoml

from import_model import BENTO_MODEL_TAG

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
        import outlines
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer and model from the BentoML model reference path
        hf_tokenizer = AutoTokenizer.from_pretrained(self.bento_model_ref.path)
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.bento_model_ref.path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda"
        )

        # Then use the loaded model with Outlines
        self.model = outlines.from_transformers(hf_model, hf_tokenizer)

    ...
```

We then need to define an HTTP endpoint using `@bentoml.api` to decorate the method `generate` of `Outlines` class:

```python
    ...

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Give me a character description.",
        json_schema: t.Optional[str] = DEFAULT_SCHEMA,
    ) -> t.Dict[str, t.Any]:
        import json
        import outlines
        from outlines.types import JsonSchema

        generator = outlines.Generator(self.model, JsonSchema(json_schema))
        character = generator(prompt)

        return json.loads(character)
```

Here `@bentoml.api` decorator defines `generate` as an HTTP endpoint that accepts a JSON request body with two fields: `prompt` and `json_schema` (optional, which allows HTTP clients to provide their own JSON schema). The type hints in the function signature will be used to validate incoming JSON requests. You can define as many HTTP endpoints as you want by using `@bentoml.api` to decorate other methods of `Outlines` class.

Now you can save the above code to `service.py` (or use [this implementation](https://github.com/dottxt-ai/outlines/blob/main/examples/bentoml/)), and run the code using the BentoML CLI.

## Run locally for testing and debugging

Then you can run a server locally by:

```shell
bentoml serve .
```

The server is now active at <http://localhost:3000>. You can interact with it using the Swagger UI or in other different ways:

<details>

<summary>CURL</summary>

```shell
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Give me a character description."
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response = client.generate(
        prompt="Give me a character description"
    )
    print(response)
```

</details>

Expected output:

```shell
{
  "name": "Aura",
  "age": 15,
  "armor": "plate",
  "weapon": "sword",
  "strength": 20
}
```

## Deploy to BentoCloud

After the Service is ready, you can deploy it to [BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/get-started.html) for better management and scalability. [Sign up](https://cloud.bentoml.com/signup) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```shell
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
