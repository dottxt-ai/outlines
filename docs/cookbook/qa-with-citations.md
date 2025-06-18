# Generate Synthetic Data and Q&A with Citations

This tutorial is adapted from the [instructor-ollama notebook](https://github.com/alonsosilvaallende/Hermes-Function-Calling/blob/main/examples/instructor_ollama.ipynb). We start with a simple example to generate synthetic data and then we approach the problem of question answering by providing citations.

We will use [llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) library. Outlines supports llama-cpp-python, but we need to install it ourselves:

```bash
pip install llama-cpp-python
```

We download the model weights by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):
```python
import llama_cpp
from outlines import generate, models

model = models.llamacpp("NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
            "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
            tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
            ),
            n_gpu_layers=-1,
            flash_attn=True,
            n_ctx=8192,
            verbose=False)
```

??? note "(Optional) Store the model weights in a custom folder"

    By default the model weights are downloaded to the hub cache but if we want so store the weights in a custom folder, we pull a quantized GGUF model [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF) by [NousResearch](https://nousresearch.com/) from [HuggingFace](https://huggingface.co/):

    ```bash
    wget https://hf.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf
    ```

    We initialize the model:

    ```python
    import llama_cpp
    from llama_cpp import Llama
    from outlines import generate, models

    llm = Llama(
        "/path/to/model/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
        ),
        n_gpu_layers=-1,
        flash_attn=True,
        n_ctx=8192,
        verbose=False
    )
    ```

## Generate Synthetic Data

We first need to define our Pydantic class for a user:

```python
from pydantic import BaseModel, Field

class UserDetail(BaseModel):
    id: int = Field(..., description="Unique identifier") # so the model keeps track of the number of users
    first_name: str
    last_name: str
    age: int
```

We then define a Pydantic class for a list of users:

```python
from typing import List

class Users(BaseModel):
    users: List[UserDetail]
```

We can use a `generate.json` by passing this Pydantic class we just defined, and call the generator:

```python
model = models.LlamaCpp(llm)
generator = generate.json(model, Users)
response = generator("Create 5 fake users", max_tokens=1024, temperature=0, seed=42)
print(response.users)
# [UserDetail(id=1, first_name='John', last_name='Doe', age=25),
# UserDetail(id=2, first_name='Jane', last_name='Doe', age=30),
# UserDetail(id=3, first_name='Bob', last_name='Smith', age=40),
# UserDetail(id=4, first_name='Alice', last_name='Smith', age=35),
# UserDetail(id=5, first_name='John', last_name='Smith', age=20)]
```

```python
for user in response.users:
    print(user.first_name)
    print(user.last_name)
    print(user.age)
    print(#####)
# John
# Doe
# 25
# #####
# Jane
# Doe
# 30
# #####
# Bob
# Smith
# 40
# #####
# Alice
# Smith
# 35
# #####
# John
# Smith
# 20
# #####
```

## QA with Citations

We first need to define our Pydantic class for QA with citations:

```python
from typing import List
from pydantic import BaseModel

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    citations: List[str]

schema = QuestionAnswer.model_json_schema()
```

We then need to adapt our prompt to the [Hermes prompt format for JSON schema](https://github.com/NousResearch/Hermes-Function-Calling?tab=readme-ov-file#prompt-format-for-json-mode--structured-outputs):

```python
def generate_hermes_prompt(question, context, schema=schema):
    return (
        "<|im_start|>system\n"
        "You are a world class AI model who answers questions in JSON with correct and exact citations "
        "extracted from the `Context`. "
        f"Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema><|im_end|>\n"
        "<|im_start|>user\n"
        + "`Context`: "
        + context
        + "\n`Question`: "
        + question + "<|im_end|>"
        + "\n<|im_start|>assistant\n"
        "<schema>"
    )
```

We can use `generate.json` by passing the Pydantic class we previously defined, and call the generator with Hermes prompt:

```python
question = "What did the author do during college?"
context = """
My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.
I went to an arts high school but in university I studied Computational Mathematics and physics.
As part of coop I worked at many companies including Stitchfix, Facebook.
I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
"""
generator = generate.json(model, QuestionAnswer)
prompt = generate_hermes_prompt(question, context)
response = generator(prompt, max_tokens=1024, temperature=0, seed=42)
print(response)
# QuestionAnswer(question='What did the author do during college?', answer='The author studied Computational Mathematics and physics in university and was also involved in starting the Data Science club, serving as its president for 2 years.', citations=['I went to an arts high school but in university I studied Computational Mathematics and physics.', 'I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.'])
```

We can do the same for a list of question-context pairs:

```python
question1 = "Where was John born?"
context1 = """
John Doe is a software engineer who was born in New York, USA.
He studied Computer Science at the Massachusetts Institute of Technology.
During his studies, he interned at Google and Microsoft.
He also founded the Artificial Intelligence club at his university and served as its president for three years.
"""

question2 = "What did Emily study in university?"
context2 = """
Emily Smith is a data scientist from London, England.
She attended the University of Cambridge where she studied Statistics and Machine Learning.
She interned at IBM and Amazon during her summer breaks.
Emily was also the head of the Women in Tech society at her university.
"""

question3 = "Which companies did Robert intern at?"
context3 = """
Robert Johnson, originally from Sydney, Australia, is a renowned cybersecurity expert.
He studied Information Systems at the University of Melbourne.
Robert interned at several cybersecurity firms including NortonLifeLock and McAfee.
He was also the leader of the Cybersecurity club at his university.
"""

question4 = "What club did Alice start at her university?"
context4 = """
Alice Williams, a native of Dublin, Ireland, is a successful web developer.
She studied Software Engineering at Trinity College Dublin.
Alice interned at several tech companies including Shopify and Squarespace.
She started the Web Development club at her university and was its president for two years.
"""

question5 = "What did Michael study in high school?"
context5 = """
Michael Brown is a game developer from Tokyo, Japan.
He attended a specialized high school where he studied Game Design.
He later attended the University of Tokyo where he studied Computer Science.
Michael interned at Sony and Nintendo during his university years.
He also started the Game Developers club at his university.
"""

for question, context in [
    (question1, context1),
    (question2, context2),
    (question3, context3),
    (question4, context4),
    (question5, context5),
]:
    final_prompt = my_final_prompt(question, context)
    generator = generate.json(model, QuestionAnswer)
    response = generator(final_prompt, max_tokens=1024, temperature=0, seed=42)
    display(question)
    display(response.answer)
    display(response.citations)
    print("\n\n")

# 'Where was John born?'
# 'John Doe was born in New York, USA.'
# ['John Doe is a software engineer who was born in New York, USA.']
#
#
# 'What did Emily study in university?'
# 'Emily studied Statistics and Machine Learning in university.'
# ['She attended the University of Cambridge where she studied Statistics and Machine Learning.']
#
#
# 'Which companies did Robert intern at?'
# 'Robert interned at NortonLifeLock and McAfee.'
# ['Robert Johnson, originally from Sydney, Australia, is a renowned cybersecurity expert. He interned at several cybersecurity firms including NortonLifeLock and McAfee.']
#
#
# 'What club did Alice start at her university?'
# 'Alice started the Web Development club at her university.'
# ['Alice Williams, a native of Dublin, Ireland, is a successful web developer. She started the Web Development club at her university and was its president for two years.']
#
#
# 'What did Michael study in high school?'
# 'Michael studied Game Design in high school.'
# ['Michael Brown is a game developer from Tokyo, Japan. He attended a specialized high school where he studied Game Design.']
```

This example was originally contributed by [Alonso Silva](https://github.com/alonsosilvaallende).
