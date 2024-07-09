# Zero-Shot Chain-of-Thought (CoT)


Zero-Shot Chain-of-Thought (CoT) is a prompting technique that encourages a language model to break down its reasoning process into steps, without providing any examples. It typically involves adding a simple phrase like "Let's approach this step by step:" or "Let's think about this logically:" before presenting the task or question. This prompts the model to generate a series of intermediate reasoning steps before arriving at the final answer, enhancing its problem-solving capabilities and providing transparency into its thought process.
    

## A worked example


To implement Zero-Shot CoT:

1. Start with your base task or question.
2. Prepend a thought-inducing phrase to encourage step-by-step reasoning. Common phrases include:
   - "Let's approach this step by step:"
   - "Let's think about this logically:"
   - "Let's break this down:"
3. Present the modified prompt to the language model.
4. Analyze the output, which should now include intermediary reasoning steps.

For example, if your original question was:
"How many eggs are left if you have a dozen eggs and use 4 for baking?"

Your Zero-Shot CoT prompt would become:
"Let's approach this step by step:
How many eggs are left if you have a dozen eggs and use 4 for baking?"

The model's response might then look like:
"1. Let's start with the given information:
   - We have a dozen eggs
   - We use 4 eggs for baking

2. First, let's recall how many eggs are in a dozen:
   - 1 dozen = 12 eggs

3. Now, we can set up a simple subtraction:
   - Initial number of eggs: 12
   - Eggs used for baking: 4
   - Remaining eggs: 12 - 4

4. Perform the subtraction:
   12 - 4 = 8

Therefore, there are 8 eggs left."

This technique allows the model to show its reasoning process, making the solution more transparent and potentially more accurate.
    
## Code Example


```python
from pydantic import BaseModel, conint
from typing import List
import outlines

class MathSolution(BaseModel):
    steps: List[str]
    answer: conint(gt=0)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, MathSolution)

prompt = """Let's approach this step by step:
What is the sum of all even numbers between 1 and 10?"""

solution = generator(prompt)
print(f"Reasoning steps:")
for step in solution.steps:
    print(f"- {step}")
print(f"\nFinal answer: {solution.answer}")
```
    

