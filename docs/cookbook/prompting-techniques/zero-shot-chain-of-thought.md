---
title: Zero-Shot Chain-of-Thought (CoT)
---

# Zero-Shot Chain-of-Thought (CoT)


Zero-Shot Chain-of-Thought (CoT) is a prompting technique that encourages a language model to break down its reasoning process into steps, without providing any examples. It uses a simple prompt instruction like "Let's approach this step by step:" or "Let's think about this logically:" before presenting the task or question. This prompts the model to generate a series of intermediate reasoning steps before arriving at the final answer, improving performance on complex reasoning tasks without needing labeled examples.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement Zero-Shot CoT:

1. Start with your task or question.
2. Prepend a thought-inducing phrase to encourage step-by-step reasoning. Common phrases include:
   - "Let's approach this step by step:"
   - "Let's think about this logically:"
   - "Let's break this down:"
3. Optionally, add an instruction to show work or explain reasoning.
4. Send the prompt to the language model.
5. Analyze the output, which should now contain a chain of reasoning steps.

For example, given a math word problem:

Original question: 
"If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

Zero-Shot CoT prompt:
"Let's approach this step by step:

If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?

Please show your work and explain your reasoning."

This prompt structure encourages the model to break down its thinking process, potentially leading to more accurate and explainable results.
    
## Code Example





```python
from typing import List
from pydantic import BaseModel

import outlines

class ReasoningStep(BaseModel):
    step_number: int
    description: str

class ChainOfThought(BaseModel):
    steps: List[ReasoningStep]
    final_answer: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
generator = outlines.generate.json(model, ChainOfThought)

prompt = """Let's approach this step by step:

If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?

Please show your work and explain your reasoning."""

reasoning = generator(prompt)
print(reasoning)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 97/97 [00:00<00:00, 
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    steps=[ReasoningStep(step_number=1, description='First, we will divide the distance traveled by the time it took to travel that distance.')] final_answer="The train's average speed is 60 miles per hour."


    
## Another Example






```python
import outlines
from pydantic import BaseModel, Field
from typing import List
from outlines.integrations.utils import convert_json_schema_to_str

model = outlines.models.transformers("google/gemma-2b")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]



```python
class ChainOfThought(BaseModel):
    reasoning_steps: List[str] = Field(..., description="List of reasoning steps")
    final_answer: str = Field(..., description="The final answer to the question")


schema_str = convert_json_schema_to_str(json_schema=ChainOfThought.model_json_schema())

prompt = f"""
Answer in json according to the following schema: {schema_str}


John has 5 apples. He gives 2 apples to his friend and then buys 3 more apples from the store. How many apples does John have now? Let's think step by step."""

generator = outlines.generate.json(model, ChainOfThought)
result = generator(prompt)

print("Reasoning steps:")
for step in result.reasoning_steps:
    print(f"- {step}")
print(f"\nFinal answer: {result.final_answer}")
```

    Compiling FSM index for all state transitions: 100%|█████████████████| 65/65 [00:02<00:00, 29.68it/s]


    Reasoning steps:
    - John has 5 apples. John gives 2 apples to his friend.
    - John has 3 apples.
    - John buys 3 apples from the store.
    
    Final answer: John now has 3 apples



```python

```
