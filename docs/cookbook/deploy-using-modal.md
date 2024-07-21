# Run Outlines using Modal

[Modal](https://modal.com/) is a serverless platform that allows you to easily run code on the cloud, including GPUs. It can come very handy for those of us who don't have a monster GPU at home and want to be able to quickly and easily provision, configure and orchestrate cloud infrastructure.

In this guide we will show you how you can use Modal to run programs written with Outlines on GPU in the cloud.

## Build the image

First we need to define our container image. We download the Mistral-7B-v0.1 model from HuggingFace as part of the definition of the image so it only needs to be done once (you need to provide an [access token](https://huggingface.co/settings/tokens))

```python
from modal import Image, App, gpu

app = App(name="outlines-app")

outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.0.37",
    "transformers==4.38.2",
    "datasets==2.18.0",
    "accelerate==0.27.2",
)

def import_model():
    import os
    os.environ["HF_TOKEN"] = "YOUR_HUGGINGFACE_TOKEN"
    import outlines
    outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

outlines_image = outlines_image.run_function(import_model)
```

We will run the JSON-structured generation example [in the README](https://github.com/outlines-dev/outlines?tab=readme-ov-file#efficient-json-generation-following-a-json-schema), with the following schema:

## Run inference

```python
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
```

To make the inference work on Modal we need to wrap the corresponding function in a `@app.function` decorator. We pass to this decorator the image and GPU on which we want this function to run (here an A100 with 80GB memory):

```python
@app.function(image=outlines_image, gpu=gpu.A100(size='80GB'))
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    import outlines

    model = outlines.models.transformers(
        "mistralai/Mistral-7B-v0.1", device="cuda"
    )

    generator = outlines.generate.json(model, schema)
    character = generator(
        f"<s>[INST]Give me a character description. Describe {prompt}.[/INST]"
    )

    print(character)
```

We then need to define a `local_entrypoint` to call our function `generate` remotely:

```python
@app.local_entrypoint()
def main(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    generate.remote(prompt)
```

Here `@app.local_entrypoint()` decorator defines `main` as the function to start from locally when running the Modal CLI. You can save above code to `example.py` (or use [this implementation](https://github.com/outlines-dev/outlines/blob/main/examples/modal_example.py)). Let's now see how to run the code on the cloud using the Modal CLI.

## Run on the cloud

First install the Modal client from PyPi:

```bash
pip install modal
```

You then need to obtain a token from Modal. To do so easily, run the following command:

```bash
modal setup
```

Once that is set you can run inference on the cloud using:

```bash
modal run example.py
```

You should see the Modal app initialize, and soon after see the result of the `print` function in your terminal. That's it!
