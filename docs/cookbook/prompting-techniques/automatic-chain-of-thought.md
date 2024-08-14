# Automatic Chain-of-Thought (Auto-CoT) Prompting


Auto-CoT is a technique that automates the process of creating Chain-of-Thought (CoT) examples for prompting. It works by first using a Zero-Shot CoT prompt on a set of questions to generate chains of thought automatically. The best-generated chains are then selected and used to construct a Few-Shot CoT prompt for the target task. This method reduces the need for manual creation of CoT examples and can potentially generate more diverse and task-specific reasoning chains.


## Step by Step Example


Let's implement Auto-CoT for basic arithmetic word problems:

1. Start with simple questions:
   Q1: "How many apples are left if you have 5 apples and eat 2?"
   Q2: "If you have $10 and spend $3 on a book, how much money do you have left?"

2. Use Zero-Shot CoT prompt to generate reasoning:
   For Q1: "Let's approach this step-by-step:
   1. We start with 5 apples.
   2. We eat 2 apples.
   3. To find how many are left, we subtract: 5 - 2 = 3
   Therefore, 3 apples are left."

   For Q2: "Let's break this down:
   1. We begin with $10.
   2. We spend $3 on a book.
   3. To find the remaining money, we subtract: $10 - $3 = $7
   So, we have $7 left."

3. Use these generated chains to create a Few-Shot CoT prompt for a slightly more complex problem:

Prompt: "Here are two examples of solving math word problems:

Q: How many apples are left if you have 5 apples and eat 2?
A: Let's approach this step-by-step:
1. We start with 5 apples.
2. We eat 2 apples.
3. To find how many are left, we subtract: 5 - 2 = 3
Therefore, 3 apples are left.

Q: If you have $10 and spend $3 on a book, how much money do you have left?
A: Let's break this down:
1. We begin with $10.
2. We spend $3 on a book.
3. To find the remaining money, we subtract: $10 - $3 = $7
So, we have $7 left.

Now, please solve this problem using the same step-by-step approach:
Q: If you have 15 candies and give 3 to your friend, then buy 4 more, how many candies do you have now?"

This prompt uses the automatically generated CoT examples to guide the model in solving a new, slightly more complex problem using the same reasoning style.

## Code Example






```python
import outlines
from outlines.integrations.utils import convert_json_schema_to_str
from pydantic import BaseModel, Field
from typing import List

model = outlines.models.transformers("meta-llama/Meta-Llama-3.1-8B", device="cuda")

class ReasoningStep(BaseModel):
    step: str = Field(..., description="A single step in the reasoning process")

class Reasoning(BaseModel):
    steps: List[ReasoningStep] = Field(..., description="List of reasoning steps")
    conclusion: str = Field(..., description="Final answer")

schema_str = convert_json_schema_to_str(Reasoning.model_json_schema())

def generate_zero_shot_cot(question: str) -> Reasoning:
    prompt = f"""Use step-by-step reasoning to solve this problem. Provide your answer in the following JSON format:
{schema_str}

Problem: {question}
"""
    generator = outlines.generate.json(model, Reasoning)
    return generator(prompt)

def solve_with_few_shot_cot(examples: List[tuple], new_question: str) -> Reasoning:
    few_shot_prompt = "Here are some examples of solving math problems step-by-step:\n\n"
    
    for q, a in examples:
        few_shot_prompt += f"Q: {q}\nA: {a.steps}\nConclusion: {a.conclusion}\n\n"
    
    few_shot_prompt += f"""Now, solve this new problem using the same step-by-step approach. Provide your answer in the following JSON format:
{schema_str}

Problem: {new_question}
"""
    generator = outlines.generate.json(model, Reasoning)
    return generator(few_shot_prompt)

# Generate Zero-Shot CoT examples
q1 = "If you have 5 apples and eat 2, how many apples do you have left?"
q2 = "If you have $10 and spend $3 on a book, how much money do you have left?"

example1 = generate_zero_shot_cot(q1)
example2 = generate_zero_shot_cot(q2)

# Use the generated examples for Few-Shot CoT
examples = [(q1, example1), (q2, example2)]
new_question = "If you have 15 candies, give 3 to your friend, then buy 4 more, how many candies do you have now?"

result = solve_with_few_shot_cot(examples, new_question)
print(result)
```


    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|██████████| 66/66 [00:06<00:00, 10.88it/s]
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
    Compiling FSM index for all state transitions: 100%|██████████| 66/66 [00:02<00:00, 26.61it/s]


    steps=[ReasoningStep(step='You have 15 candies'), ReasoningStep(step='You give 3 to your friend'), ReasoningStep(step='So now you have 15-3 = 12'), ReasoningStep(step='You buy 4 more candies')] conclusion='Now you have 12+4 = 16 candies'



```python

```
