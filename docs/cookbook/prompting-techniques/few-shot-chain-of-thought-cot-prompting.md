# Few-Shot Chain-of-Thought (CoT) Prompting


Few-Shot Chain-of-Thought (CoT) prompting is an advanced technique that combines few-shot learning with chain-of-thought reasoning. This method involves providing the language model with a few examples of problems and their solutions, including the step-by-step reasoning process (chain of thought) for each example. By demonstrating the problem-solving process, the model is encouraged to mimic this approach when tackling new, similar problems.

The key components of Few-Shot CoT prompting are:
1. A set of example problems (usually 2-5)
2. Step-by-step reasoning for each example problem
3. The correct answer for each example
4. A new problem for the model to solve

This technique helps the model understand not just the correct answers, but also the reasoning process to arrive at those answers, leading to improved performance on complex reasoning tasks.
    

## A worked example


Here's how to implement Few-Shot CoT prompting:

1. Prepare your prompt:
   a. Start with a brief instruction explaining the task.
   b. Provide 2-3 example problems with their step-by-step solutions and final answers.
   c. Present the new problem you want the model to solve.

2. Example prompt structure:
```
Solve the following math word problems step by step. Show your reasoning for each step.

Example 1:
Problem: John has 5 apples. He gives 2 to his friend and buys 3 more. How many apples does John have now?
Solution:
Step 1: Understand the initial number of apples John has.
John starts with 5 apples.

Step 2: Calculate how many apples John has after giving some away.
5 apples - 2 apples = 3 apples

Step 3: Add the number of apples John bought.
3 apples + 3 apples = 6 apples

Therefore, John now has 6 apples.

Example 2:
Problem: A store sells shirts for $25 each. If they offer a 20% discount, what is the final price of one shirt?
Solution:
Step 1: Calculate the discount amount.
20% of $25 = 0.20 × $25 = $5

Step 2: Subtract the discount from the original price.
$25 - $5 = $20

Therefore, the final price of one shirt is $20.

Now, solve this problem:
A bakery sold 136 cupcakes on Monday and 25% more on Tuesday. How many cupcakes did they sell in total over these two days?
```

3. Submit this prompt to the language model.

4. The model should now provide a step-by-step solution for the new problem, following the pattern established in the examples.

5. Review the model's response to ensure it has followed the chain-of-thought reasoning process and provided a clear, logical solution.

By using this technique, you're guiding the model to break down complex problems into smaller, manageable steps, leading to more accurate and explainable results.
    
## Code Example


```python
from typing import List
from pydantic import BaseModel, Field

import outlines

class Step(BaseModel):
    description: str
    calculation: str

class Solution(BaseModel):
    steps: List[Step]
    final_answer: int

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, Solution)

prompt = """Solve the following math word problems step by step. Show your reasoning for each step.

Example 1:
Problem: John has 5 apples. He gives 2 to his friend and buys 3 more. How many apples does John have now?
Solution:
Step 1: Understand the initial number of apples John has.
John starts with 5 apples.

Step 2: Calculate how many apples John has after giving some away.
5 apples - 2 apples = 3 apples

Step 3: Add the number of apples John bought.
3 apples + 3 apples = 6 apples

Therefore, John now has 6 apples.

Example 2:
Problem: A store sells shirts for $25 each. If they offer a 20% discount, what is the final price of one shirt?
Solution:
Step 1: Calculate the discount amount.
20% of $25 = 0.20 × $25 = $5

Step 2: Subtract the discount from the original price.
$25 - $5 = $20

Therefore, the final price of one shirt is $20.

Now, solve this problem:
A bakery sold 136 cupcakes on Monday and 25% more on Tuesday. How many cupcakes did they sell in total over these two days?
"""

solution = generator(prompt)
print(solution)
```
    

