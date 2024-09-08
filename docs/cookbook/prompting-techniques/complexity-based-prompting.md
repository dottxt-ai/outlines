---
title: Complexity-based Prompting
---

# Complexity-based Prompting


[Complexity-based Prompting](https://openreview.net/forum?id=yf1icZHC-l9) is an advanced technique that enhances Chain-of-Thought (CoT) prompting by focusing on complex examples and leveraging multiple reasoning paths. It involves two key steps:

1. Prompt Creation: Select and annotate complex examples based on factors like question length or required reasoning steps. These examples are used to build the prompt.

2. Inference: For each new problem, generate multiple reasoning chains (answers). Then, apply a majority voting system among chains that exceed a certain length threshold. This is based on the assumption that longer reasoning indicates higher answer quality.

This technique aims to improve performance on challenging tasks by exposing the model to more complex reasoning patterns and aggregating multiple solution attempts.


## A worked example


Let's apply Complexity-based Prompting to solve math word problems:

1. Select complex examples:
   Simple: "John has 5 apples and gives 2 to Sarah. How many does he have left?"
   Complex: "A store sells notebooks for $2.50 each. If you buy 3 notebooks and have a 20% off coupon, how much will you pay after tax if the tax rate is 8%?"

   We'll use the complex example in our prompt.

2. Create the prompt:
   "Solve this math problem step by step:
   Q: A store sells notebooks for $2.50 each. If you buy 3 notebooks and have a 20% off coupon, how much will you pay after tax if the tax rate is 8%?
   A: Let's break this down:
   1. Cost of notebooks: $2.50 x 3 = $7.50
   2. Apply 20% discount: $7.50 x 0.80 = $6.00
   3. Calculate tax: $6.00 x 0.08 = $0.48
   4. Total cost: $6.00 + $0.48 = $6.48
   Therefore, you will pay $6.48 after tax.

   Now solve this problem:
   Q: If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

3. Generate multiple reasoning chains (e.g., 3):
   Chain 1:
   1. We need to find the speed, which is distance divided by time.
   2. Distance = 120 miles
   3. Time = 2 hours
   4. Speed = 120 miles ÷ 2 hours = 60 miles per hour

   Chain 2:
   1. Speed is calculated by dividing distance by time.
   2. We have a distance of 120 miles and a time of 2 hours.
   3. 120 ÷ 2 = 60
   4. The average speed is 60 miles per hour.

   Chain 3:
   1. To find average speed, use the formula: Speed = Distance / Time
   2. Distance traveled = 120 miles
   3. Time taken = 2 hours
   4. Speed = 120 / 2 = 60 mph

4. Apply majority voting:
   All chains exceed our length threshold (e.g., 3 steps) and arrive at the same answer.
   The final answer is 60 miles per hour.

This example demonstrates how Complexity-based Prompting uses a complex example in the prompt and leverages multiple reasoning chains to arrive at a robust solution.

## Code Example






```python
import outlines
from pydantic import BaseModel, Field
from typing import List

# Define Pydantic models
class MathProblem(BaseModel):
    question: str
    correct_answer: float

class ReasoningStep(BaseModel):
    step: str = Field(..., description="A single step in the reasoning process")

class ReasoningChain(BaseModel):
    steps: List[ReasoningStep]
    final_answer: float

# Initialize the model
model = outlines.models.transformers("google/gemma-2b")

# Create the prompt
prompt = f"""Solve this math problem step by step:
Q: A store sells notebooks for $2.50 each. If you buy 3 notebooks and have a 20% off coupon, how much will you pay after tax if the tax rate is 8%?
A: Let's break this down:
1. Cost of notebooks: $2.50 x 3 = $7.50
2. Apply 20% discount: $7.50 x 0.80 = $6.00
3. Calculate tax: $6.00 x 0.08 = $0.48
4. Total cost: $6.00 + $0.48 = $6.48
Therefore, you will pay $6.48 after tax.

Now solve this problem step by step:
Q: If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?
"""

# Generate multiple reasoning chains
generator = outlines.generate.json(model, ReasoningChain)
chains = [generator(prompt) for _ in range(3)]

# Implement majority voting
def majority_vote(chains, threshold=3):
    valid_chains = [chain for chain in chains if len(chain.steps) >= threshold]
    if not valid_chains:
        return None
    answers = [chain.final_answer for chain in valid_chains]
    print(answers)
    return max(set(answers), key=answers.count)

final_answer = majority_vote(chains)
print(f"The final answer based on majority voting is: {final_answer} miles per hour")
```


    [220.4, 14.834, 30.0]
    The final answer based on majority voting is: 220.4 miles per hour


The answer is incorrect. We might want to try multiple generations, a larger model with better reasoning, or a different technique.
