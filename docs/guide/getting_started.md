---
title: Getting Started
---

# Getting Started

## Installation

We recommend using `uv` to install Outlines. You can find `uv` installation instructions [here](https://github.com/astral-sh/uv).

```shell
uv pip install 'outlines[transformers]'
```

or the classic `pip`:

```shell
pip install 'outlines[transformers]'
```

For more information, see the [installation guide](./installation).

## Creating a Model

Outlines contains a variety of models that wrap LLM inference engines/clients. For each of them, you need to install the model's associated library as described in the [installation guide](../installation).

The full list of available models along with detailed explanation on how to use them can be found in the [models page](../features/models/index.md) of the Features section of the documentation.

For a quick start, you can find below an example of how to initialize all supported models in Outlines:

=== "vLLM"

    ```python
    import outlines
    from openai import OpenAI

    # You must have a separate vLLM server running
    # Create an OpenAI client with the base URL of the VLLM server
    openai_client = OpenAI(base_url="http://localhost:11434/v1")

    # Create an Outlines model
    model = outlines.from_vllm(openai_client, "microsoft/Phi-3-mini-4k-instruct")
    ```

=== "Ollama"

    ```python
    import outlines
    from ollama import Client

    # Create an Ollama client
    ollama_client = Client()

    # Create an Outlines model, the model must be available on your system
    model = outlines.from_ollama(ollama_client, "tinyllama")
    ```

=== "OpenAI"

    ```python
    import outlines
    from openai import OpenAI

    # Create an OpenAI client instance
    openai_client = OpenAI()

    # Create an Outlines model
    model = outlines.from_openai(openai_client, "gpt-4o")
    ```

=== "Transformers"

    ```python
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Define the model you want to use
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Create a HuggingFace model and tokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create an Outlines model
    model = outlines.from_transformers(hf_model, hf_tokenizer)
    ```


=== "llama.cpp"

    ```python
    import outlines
    from llama_cpp import Llama

    # Model to use, it will be downloaded from the HuggingFace hub
    repo_id = "TheBloke/Llama-2-13B-chat-GGUF"
    file_name = "llama-2-13b-chat.Q4_K_M.gguf"

    # Create a Llama.cpp model
    llama_cpp_model = Llama.from_pretrained(repo_id, file_name)

    # Create an Outlines model
    model = outlines.from_llamacpp(llama_cpp_model)
    ```

=== "Gemini"

    ```python
    import outlines
    from google.generativeai import GenerativeModel

    # Create a Gemini client
    gemini_client = GenerativeModel()

    # Create an Outlines model
    model = outlines.from_gemini(gemini_client, "gemini-1-5-flash")
    ```

=== "mlx-lm"

    ```python
    import outlines
    import mlx_lm

    # Create an MLXLM model with the output of mlx_lm.load
    # The model will be downloaded from the HuggingFace hub
    model = outlines.from_mlxlm(
        **mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
    )
    ```

=== "SgLang"

    ```python
    import outlines
    from openai import OpenAI

    # You must have a separate SgLang server running
    # Create an OpenAI client with the base URL of the SgLang server
    openai_client = OpenAI(base_url="http://localhost:11434/v1")

    # Create an Outlines model
    model = outlines.from_sglang(openai_client)
    ```

=== "TGI"

    ```python
    # SgLang

    import outlines
    from huggingface_hub import InferenceClient

    # You must have a separate TGI server running
    # Create an InferenceClient client with the base URL of the TGI server
    tgi_client = InferenceClient("http://localhost:8080")

    # Create an Outlines model
    model = outlines.from_tgi(tgi_client)
    ```

=== "vLLM (offline)"

    ```python
    import outlines
    from vllm import LLM

    # Create a vLLM model
    vllm_model = LLM("microsoft/Phi-3-mini-4k-instruct")

    # Create an Outlines model
    model = outlines.from_vllm_offline(vllm_model)
    ```


## Generating Text

Once you have created the Outlines model for your inference engine/client, you are already all set to generate text! Models are callable such that you can simply call them with a text prompt. For instance:

```python
model = <your_model_as_defined_above>

# Call the model to generate text
result = model("Write a short story about a cat.")
print(result) # 'In a quiet village where the cobblestones hummed softly beneath the morning mist...'
```

Most models also support streaming through the use of a `streaming` method. You can directly use with a prompt just like regular text generation. For instance:

```python
model = <your_model_as_defined_above>

# Stream text
for chunk in model.streaming("Write a short story about a cat.")
    print(chunk) # 'In ...'
```

## Structured Generation

Outlines follows a simple pattern that mirrors Python's own type system for structured outputs. Simply specify the desired output type as you would when using type hinting with a function, and Outlines will ensure your data matches that structure exactly.

Supported output types can be organized in 5 categories:

- [Basic Types](../../features/core/output_types#basic-python-types): `int`, `float`, `bool`...
- [Multiple Choices](../../features/core/output_types#multiple-choices): using `Literal` or `Enum`
- [JSON Schemas](../../features/core/output_types#json-schemas): using a wide range of possible objects including Pydantic models and dataclasses
- [Regex](../../features/core/output_types#regex-patterns): through the Outlines's `Regex` object
- [Context-free Grammars](../../features/core/output_types#context-free-grammars): through the Outlines's `CFG` object

Consult the section on [Output Types](../../features/core/output_types.md) in the features documentation for more detailed information on all supported types for each output type category.

In the meantime, you can find below examples of using each of the five output type categories:

=== "Basic Types"

    ```python
    model = <your_model_as_defined_above>

    # Generate an integer
    result = model("How many countries are there in the world?", int)
    print(result) # '200'
    ```

=== "Multiple Choice"

    ```python
    from enum import Enum

    # Define our multiple choice output type
    class PizzaOrBurger(Enum):
        pizza = "pizza"
        burger = "burger"

    model = <your_model_as_defined_above>

    # Generate text corresponding to either of the choices defined above
    result = model("What do you want to eat, a pizza or a burger?", PizzaOrBurger)
    print(result) # 'pizza'
    ```

=== "JSON Schemas"

    ```python
    from datetime import date
    from typing import Dict, List, Union
    from pydantic import BaseModel

    model = <your_model_as_defined_above>

    # Define the class we will use as an output type
    class Character(BaseModel):
        name: str
        birth_date: date
        skills: Union[Dict, List[str]]

    # Generate a character
    result = model("Create a character", Character)
    print(result) # '{"name": "Aurora", "birth_date": "1990-06-15", "skills": ["Stealth", "Diplomacy"]}'
    print(Character.model_validate_json(result)) # name=Aurora birth_date=datetime.date(1990, 6, 15) skills=['Stealth', 'Diplomacy']
    ```

=== "Regex"

    ```python
    from outlines.types import Regex

    model = <your_model_as_defined_above>

    # Define our regex for a 3 digit number
    output_type = Regex(r"[0-9]{3}")

    # Generate the number
    result = model("Write a 3 digit number", output_type)
    print(result) # '236'
    ```

=== "Context-free Grammars"

    ```python
    from outlines.types import CFG

    model = <your_model_as_defined_above>

    # Define your Lark grammar as string
    arithmetic_grammar = """
        ?start: sum

        ?sum: product
            | sum "+" product   -> add
            | sum "-" product   -> sub

        ?product: atom
            | product "*" atom  -> mul
            | product "/" atom  -> div

        ?atom: NUMBER           -> number
            | "-" atom         -> neg
            | "(" sum ")"

        %import common.NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """

    # Generate an arithmetic operation
    result = model("Write an arithmetic operation", CFG(grammar_string))
    print(result) # '2 + 3'
    ```

It's important to note that not all output types are available for all models due to limitations in the underlying inference engines. The [Models](../features/models/index.md) section of the features documentation includes a features matrix that summarize the availability of output types.

## Generators

Generators are an important type of objects in Outlines that are used to encapsulate a model and an output type. After having created a generator, you can call it using a similar interface to a model and it will generate text conforming to the output type you initially provided.

This feature is useful if you want to generate text several times for given model and output type. Not only does it prevent having to include the same output type at each call, but it also allows us to compile the output type only once instead of doing it at each generation (which is important for local models as this operation can be expensive).

For instance:

```python
from typing import Literal
from outlines import Generator

model = <your_model_as_defined_above>

# Create a generator
generator = Generator(model, Literal["pizza", "burger"])

# Call it as you would call a model
result = generator("What do you want to eat, a pizza or a burger?")
print(result) # pizza
```

You can find more information on generators in the dedicated page on [Generators](../features/core/generator.md) in the features documentation.

## Other features

On top of more detailed explanation on the concepts already discussed here, the [Features](../features/index.md) section of the documentation contains information on additional Outlines features such as applications, prompt templates, the regex DSL...
