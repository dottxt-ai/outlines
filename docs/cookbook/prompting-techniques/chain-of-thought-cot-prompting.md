# Chain-of-Thought (CoT) Prompting


Chain-of-Thought (CoT) Prompting is a technique that encourages language models to articulate their reasoning process step-by-step before providing a final answer. It typically uses few-shot prompting, where the prompt includes examples of questions along with detailed reasoning paths and correct answers. This approach helps the model break down complex problems into smaller, more manageable steps, often leading to improved performance on tasks requiring multi-step reasoning, such as mathematical problems or logical deductions.
    

## A worked example


To implement Chain-of-Thought (CoT) Prompting:

1. Prepare your prompt with one or more examples that demonstrate the desired reasoning process. Each example should include:
   - A question
   - A step-by-step reasoning path
   - The correct answer

2. Add your actual question at the end of the prompt.

3. Submit the entire prompt to the language model.

Here's a specific example:

Prompt:
"""
Q: Sarah has 3 apples. She gives 1 apple to her friend and buys 2 more from the store. How many apples does Sarah have now?
A: Let's think through this step-by-step:
1. Sarah starts with 3 apples.
2. She gives away 1 apple, so now she has 3 - 1 = 2 apples.
3. She then buys 2 more apples from the store.
4. Now she has 2 (from step 2) + 2 (newly bought) = 4 apples.
Therefore, Sarah now has 4 apples.

Q: A train travels 120 miles in 2 hours. What is its average speed in miles per hour?
A:
"""

When you submit this prompt to the language model, it should respond with a step-by-step reasoning process similar to the example, followed by the final answer for the train speed question.
    
## Code Example


```python
from typing import List
from pydantic import BaseModel

import outlines

class MathSolution(BaseModel):
    steps: List[str]
    final_answer: float

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

prompt = """You are a math tutor who explains solutions step-by-step.

Example:
Q: A train travels 120 miles in 2 hours. What is its average speed in miles per hour?
A: Let's solve this step-by-step:
1. We know the train traveled 120 miles.
2. The journey took 2 hours.
3. To find the average speed, we divide the distance by the time.
4. Average speed = 120 miles ÷ 2 hours
5. 120 ÷ 2 = 60
Therefore, the train's average speed is 60 miles per hour.

Now, solve this problem:
Q: If a car travels 280 miles in 4 hours, what is its average speed in miles per hour?
A:
"""

generator = outlines.generate.json(model, MathSolution)
solution = generator(prompt)

print("Step-by-step solution:")
for i, step in enumerate(solution.steps, 1):
    print(f"{i}. {step}")
print(f"\nFinal answer: {solution.final_answer} miles per hour")
```
    

