# Build perspective-taking agents with SimToM

Prompting strategies like Chain-of-Thought (CoT) can improve LLMs' reasoning capabilities. However, they underwhelm in tasks that require keeping track of inconsistent world states. [SimToM](https://arxiv.org/abs/2311.10227) proposes a simple, two-stage prompting framework for LLMs inspired by Simulation Theory. The authors showed that this approach outperforms zero-shot prompting and CoT on ToMI and BigToM, two benchmarks with Theory of Mind questions.

In this example, we will implement SimToM with a few lines of code using Outlines' prompt templating and structured generation capabilities.

## How SimToM works

SimToM calls an LLM with two consecutive prompts:

1. **Perspective-taking**: The first prompt receives a `story` and a `character`. The goal is to understand the situation based on the character's point of view and filter out the rest of the story.
2. **Question-Answering**: The second prompt receives the character's point of view from the previous step and tasks the LLM to answer a question using that context.

![Figure 2 in the paper](./images/simtom.png)

## Outlines implementation

To implement SimToM with Outlines, we will need to:

1. Write the prompts with [prompt templates](https://dottxt-ai.github.io/outlines/latest/reference/prompting/).
2. Define the JSON object each prompt will return using Pydantic.
3. Generate responses with a Mistral model using the [transformers integration](https://dottxt-ai.github.io/outlines/latest/reference/models/transformers/).

Let's dive into it!

### Using Prompt Templates

The authors have shared their code, prompts and data in [this GitHub repository](https://github.com/shawnsihyunlee/simulatedtom). Below, we define in Outlines the prompts they used for the ToMI dataset:

```python
from outlines import Template

perspective_taking = Template.from_file("prompt_templates/simtom_prospective_taking.txt")
simulation = Template.from_file("prompt_templates/simtom_simulation.txt")
```

### JSON Structured Generation

Outlines guarantees that the LLM will return a valid JSON object, which we can specify as a Pydantic model.

We will need two Pydantic models for SimToM, one for each prompt:

```python
from pydantic import BaseModel, Field
from typing import List

class PerspectiveTaking(BaseModel):
    """This is for the first prompt."""
    character: str = Field(description="The character we extract the events for.")
    events: List[str] = Field(description="All events that the character knows about.")

class Simulation(BaseModel):
    """This is for the second prompt."""
    answer: str
```

### Calling an LLM

Let's try SimToM with an example from the ToMI dataset:

```python
story = """
1 Aria entered the front_yard.
2 Aiden entered the front_yard.
3 The grapefruit is in the green_bucket.
4 Aria moved the grapefruit to the blue_container.
5 Aiden exited the front_yard.
6 Noah entered the playroom.
"""
question = "7 Where was the grapefruit at the beginning?"
character = "Aria"
```

We load `Mistral-7B-Instruct-v0.3`, create the prompt using the template we defined earlier, and generate a structured response. As a reminder, the goal of the first call is to get all the events a character, `Aria`, knows about.

```python
import transformers
import outlines
# Load an LLM from Hugging Face
MODEL_NAME = "mistral-community/Mistral-7B-Instruct-v0.3"
model = outlines.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME),
    transformers.AutoTokenizer.from_pretrained(MODEL_NAME),
)

perspective_prompt = perspective_taking(story=story, character=character)

# Call Mistral 7B with the first prompt
generator = outlines.Generator(model, PerspectiveTaking)
perspective = generator(perspective_prompt, max_new_tokens=1024)

print(perspective)
# {'character': 'Aria', 'events': ['1 Aria entered the front_yard.', '3 The grapefruit is in the green_bucket.', '4 Aria moved the grapefruit to the blue_container.']}
```

Not bad! We will now generate the second prompt with those events.

```python
import json

sim_prompt = simulation(events=json.loads(perspective)["events"], name=character, question=question)

# Call Mistral 7B with the second prompt
generator = outlines.Generator(model, Simulation)
result = generator(sim_prompt, max_new_tokens=1024)

print(result)
# {'answer': 'green_bucket'}
```

And this is it! SimToM could be useful in agentic workflows, where agents must act based on what they know, not all available information. One caveat of SimToM is that the perspective-taking step may remove important information, leading to wrong results. As the authors note in their paper, it can feature as a simple and effective baseline for evaluating LLMs on Theory of Mind reasoning tasks.
