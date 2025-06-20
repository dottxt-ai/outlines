# Run Outlines using Modal

[Modal](https://modal.com/) is a serverless platform that allows you to easily run code on the cloud, including GPUs. It can come very handy for those of us who don't have a monster GPU at home and want to be able to quickly and easily provision, configure and orchestrate cloud infrastructure.

In this guide we will show you how you can use Modal to run programs written with Outlines on GPU in the cloud.

## Requirements

We recommend installing `modal` and `outlines` in a virtual environment. You can create one with:

```shell
python -m venv venv
source venv/bin/activate
```

Then install the required packages:

```shell
pip install modal outlines
```

## Build the image

First we need to define our container image. If you need to access a gated model, you will need to provide an [access token](https://huggingface.co/settings/tokens). See the `.env` call below for how to provide a HuggingFace token.

Setting a token is best done by setting an environment variable `HF_TOKEN` with your token. If you do not wish to do this, we provide a commented-out line in the code to set the token directly in the code.

```python
from modal import Image, App, gpu
import os

# This creates a modal App object. Here we set the name to "outlines-app".
# There are other optional parameters like modal secrets, schedules, etc.
# See the documentation here: https://modal.com/docs/reference/modal.App
app = App(name="outlines-app")

# Specify a language model to use.
# Another good model to use is "NousResearch/Hermes-2-Pro-Mistral-7B"
language_model = "mistral-community/Mistral-7B-v0.2"

# Please set an environment variable HF_TOKEN with your Hugging Face API token.
# The code below (the .env({...}) part) will copy the token from your local
# environment to the container.
# More info on Image here: https://modal.com/docs/reference/modal.Image
outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines",
    "transformers",
    "datasets",
    "accelerate",
    "sentencepiece",
).env({
    # This will pull in your HF_TOKEN environment variable if you have one.
    'HF_TOKEN':os.environ['HF_TOKEN']

    # To set the token directly in the code, uncomment the line below and replace
    # 'YOUR_TOKEN' with the HuggingFace access token.
    # 'HF_TOKEN':'YOUR_TOKEN'
})
```

## Setting the container up

When running longer Modal apps, it's recommended to download your language model when the container starts, rather than when the function is called. This will cache the model for future runs.

```python
# This function imports the model from Hugging Face. The modal container
# will call this function when it starts up. This is useful for
# downloading models, setting up environment variables, etc.
def import_model():
    import outlines
    import transformers

    outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(language_model),
        transformers.AutoTokenizer.from_pretrained(language_model)
    )

# This line tells the container to run the import_model function when it starts.
outlines_image = outlines_image.run_function(import_model)
```

## Define a schema

We will run the JSON-structured generation example [in the README](https://github.com/dottxt-ai/outlines?tab=readme-ov-file#efficient-json-generation-following-a-json-schema), with the following schema:

```python
# Specify a schema for the character description. In this case,
# we want to generate a character with a name, age, armor, weapon, and strength.
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

To make the inference work on Modal we need to wrap the corresponding function in a `@app.function` decorator. We pass to this decorator the image and GPU on which we want this function to run.

Let's choose an A100 with 80GB memory. Valid GPUs can be found [here](https://modal.com/docs/reference/modal.gpu).

```python
# Define a function that uses the image we chose, and specify the GPU
# and memory we want to use.
@app.function(image=outlines_image, gpu=gpu.A100(size='80GB'))
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    # Remember, this function is being executed in the container,
    # so we need to import the necessary libraries here. You should
    # do this with any other libraries you might need.
    import outlines
    import transformers
    from outlines.types import JsonSchema

    # Load the model into memory. The import_model function above
    # should have already downloaded the model, so this call
    # only loads the model into GPU memory.
    outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(language_model, device_map="cuda"),
        transformers.AutoTokenizer.from_pretrained(language_model)
    )

    # Generate a character description based on the prompt.
    # We use the .json generation method -- we provide the
    # - model: the model we loaded above
    # - schema: the JSON schema we defined above
    generator = outlines.Generator(model, JsonSchema(schema))

    # Make sure you wrap your prompt in instruction tags ([INST] and [/INST])
    # to indicate that the prompt is an instruction. Instruction tags can vary
    # by models, so make sure to check the model's documentation.
    character = generator(
        f"<s>[INST]Give me a character description. Describe {prompt}.[/INST]"
    )

    # Print out the generated character.
    print(character)
```

We then need to define a `local_entrypoint` to call our function `generate` remotely.

```python
@app.local_entrypoint()
def main(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    # We use the "generate" function defined above -- note too that we are calling
    # .remote() on the function. This tells modal to run the function in our cloud
    # machine. If you want to run the function locally, you can call .local() instead,
    # though this will require additional setup.
    generate.remote(prompt)
```

Here `@app.local_entrypoint()` decorator defines `main` as the function to start from locally when using the Modal CLI. You can save above code to `example.py` (or use [this implementation](https://github.com/dottxt-ai/outlines/blob/main/examples/modal_example.py)). Let's now see how to run the code on the cloud using the Modal CLI.

## Run on the cloud

First install the Modal client from PyPi, if you have not already:

```shell
pip install modal
```

You then need to obtain a token from Modal. Run the following command:

```shell
modal setup
```

Once that is set you can run inference on the cloud using:

```shell
modal run example.py
```

You should see the Modal app initialize, and soon after see the result of the `print` function in your terminal. That's it!
