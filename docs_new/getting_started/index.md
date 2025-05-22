---
title: Getting Started
---

# Getting Started

## Installation

We recommend using `uv` to install Outlines. You can find `uv` installation instructions [here](https://github.com/astral-sh/uv).

To install Outlines with the `transformers` backend, run:

```bash
uv pip install 'outlines[transformers]'
```

or the classic `pip`:

```bash
pip install 'outlines[transformers]'
```

For other backends, see the [installation guide](../installation).

## Quick start

Outlines wraps around a variety of LLM inference backends, described in the [installation guide](../installation). The following example shows how to use Outlines with HuggingFace's [`transformers` library](https://huggingface.co/docs/transformers/en/index).

=== "Transformers"

    ```python
    import outlines
    from transformers
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        name: str
        age: int

    # Create an Outlines model using transformers
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(model_name),
        transformers.AutoTokenizer.from_pretrained(model_name),
    )

    # Generate a response
    response = model("Create a character.", Person) # { "name": "John", "age": 30 }
    ```

=== "OpenAI"

    ```python
    import outlines
    from openai import OpenAI
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        name: str
        age: int

    # Create an OpenAI client instance
    openai_client = OpenAI()

    # Create an Outlines model
    model = outlines.from_openai(openai_client, "gpt-4o")

    # Generate a response
    response = model("Create a character.", Person) # { "name": "John", "age": 30 }
    ```

=== "vLLM (offline)"

    ```python
    import outlines
    from pydantic import BaseModel
    from vllm import LLM

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        role: str
        age: int

    # Model to use, it will be downloaded from the HuggingFace hub
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    # Create a vLLM model
    vllm_model = LLM(model_id)

    # Create an Outlines model
    model = outlines.from_vllm_offline(vllm_model)

    # Generate a response
    response = model("Create a character.", Person) # { "name": "John", "age": 30 }
    ```

=== "vLLM (online)"

    ```python
    import outlines
    from openai import OpenAI
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        name: str
        age: int

    # You must have a separete vLLM server running
    # Create an OpenAI client with the base URL of the VLLM server
    openai_client = OpenAI(base_url="http://localhost:11434/v1")

    # Specify the model available on the VLLM server to use
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    # Create an Outlines model
    model = outlines.from_vllm(openai_client, model_id)

    # Generate a response
    response = model("Create a character.", Person)
    # { "name": "John", "age": 30 }
    ```

=== "llama.cpp"

    ```python
    import outlines
    from llama_cpp import Llama
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        name: str
        age: int

    # Model to use, it will be downloaded from the HuggingFace hub
    repo_id = "TheBloke/Llama-2-13B-chat-GGUF"
    file_name = "llama-2-13b-chat.Q4_K_M.gguf"

    # Create a Llama.cpp model
    llama_cpp_model = Llama.from_pretrained(repo_id, file_name)

    # Create an Outlines model
    model = outlines.from_llamacpp(llama_cpp_model)

    # Generate a response
    response = model("Create a character.", Person) # { "name": "John", "age": 30 }
    ```

=== "Dottxt"

    ```python
    import outlines
    from dottxt.client import Dottxt
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        name: str
        age: int

    # Create an Dottxt client
    client = Dottxt()

    # Create an Outlines model
    model = outlines.from_dottxt(client)

    # Generate a response
    response = model("Create a character.", Person) # { "name": "John", "age": 30 }
    ```

=== "Anthropic"

    ```python
    import outlines
    from anthropic import Anthropic
    from pydantic import BaseModel

    # Create an Anthropic client
    client = Anthropic()

    # Create an Outlines model
    model = outlines.from_anthropic(client, "claude-3-haiku-20240307")

    # Generate a response
    response = model("Create a character.", max_tokens=20) # Here is a character I have created:\n\nName: Ayla Samara
    ```

=== "Ollama"

    ```python
    import outlines
    from ollama import Client
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        name: str
        age: int

    # Create an Ollama client
    client = Client()

    # Create an Outlines model, the model must be available on your system
    model = outlines.from_ollama(client, "tinyllama")

    # Generate a response
    response = model("Create a character.", Person)
    # { "name": "John", "age": 30 }
    ```

=== "mlx-lm"

    ```python
    import outlines
    import mlx_lm
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        role: str
        age: int

    # Create an MLXLM model with the output of mlx_lm.load
    # The model will be downloaded from the HuggingFace hub
    model = outlines.from_mlxlm(mlx_lm.load(
        "mlx-community/SmolLM-135M-Instruct-4bit"
    ))

    # Generate a response
    response = model("Create a character.", Person)
    # { "name": "John", "age": 30 }
    ```

=== "Gemini"

    ```python
    import outlines
    from google.generativeai import GenerativeModel
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        name: str
        age: int

    # Create a Gemini client
    client = GenerativeModel()

    # Create an Outlines model
    model = outlines.from_gemini(client)

    # Generate a response
    response = model("Create a character.", Person) # { "name": "John", "age": 30 }
    ```

=== "SGLang"

    ```python
    # SGLang

    import outlines
    from openai import OpenAI
    from pydantic import BaseModel

    # Define a Pydantic model describing the output format
    class Person(BaseModel):
        # Set name and age fields to string and int respectively
        name: str
        age: int

    # You must have a separete SGLang server running
    # Create an OpenAI client with the base URL of the SGLang server
    openai_client = OpenAI(base_url="http://localhost:11434/v1")

    # Create an Outlines model
    model = outlines.from_sglang(openai_client)

    # Generate a response
    response = model("Create a character.", Person) # { "name": "John", "age": 30 }
    ```

### JSON

```python

# Define the input text
person_text = """
John Doe
30
john.doe@example.com
"""

# Apply chat templating to the input text
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": """
        You are a master of extracting information from text.
        """},
        {"role": "user", "content": person_text}
    ],
    tokenize=False
)

# Generate the output
result = model(
    prompt,
    Person,

    # Note: transformers has an extremely small default
    # max_new_tokens, which is often not enough for the
    # full JSON output. You will experience errors if your
    # model is unable to generate the full JSON output due
    # to the max_new_tokens limit.
    max_new_tokens=100
)

print(result)
```

Result:

```json
{
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com"
}
```

### Regex

> [!NOTE]
> Insert regex example here

### Multiple choice

> [!NOTE]
> Insert multiple choice example here
