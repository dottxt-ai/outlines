# Contrastive CoT Prompting


Contrastive CoT (Chain-of-Thought) Prompting is an advanced prompting technique that enhances the standard Chain-of-Thought approach. This method involves providing the language model with both correct and incorrect reasoning examples for solving problems. By showcasing both proper problem-solving steps and common pitfalls or mistakes, the technique aims to improve the model's ability to distinguish between valid and invalid reasoning processes. This approach has shown significant improvements in areas such as Arithmetic Reasoning and Factual QA, as it helps the model to better understand what constitutes good reasoning and what to avoid.
    

## A worked example


Here's a step-by-step guide to implement Contrastive CoT Prompting:

1. Choose a problem type: For this example, we'll use a simple arithmetic word problem.

2. Create a correct chain of thought:
Problem: If Sarah has 5 apples and gives 2 to her friend, how many apples does she have left?
Correct reasoning: 
- Sarah starts with 5 apples
- She gives away 2 apples
- To find the remaining apples, we subtract: 5 - 2 = 3
- Therefore, Sarah has 3 apples left

3. Create an incorrect chain of thought:
Incorrect reasoning:
- Sarah starts with 5 apples
- She gives away 2 apples
- To find the remaining apples, we add: 5 + 2 = 7
- Therefore, Sarah has 7 apples left

4. Formulate the prompt with both examples:
Here's how to solve arithmetic word problems:

Correct example:
Problem: If Sarah has 5 apples and gives 2 to her friend, how many apples does she have left?
Reasoning: 
- Sarah starts with 5 apples
- She gives away 2 apples
- To find the remaining apples, we subtract: 5 - 2 = 3
- Therefore, Sarah has 3 apples left
This reasoning is correct because we subtract the number of apples given away from the initial number.

Incorrect example:
Problem: If Sarah has 5 apples and gives 2 to her friend, how many apples does she have left?
Reasoning:
- Sarah starts with 5 apples
- She gives away 2 apples
- To find the remaining apples, we add: 5 + 2 = 7
- Therefore, Sarah has 7 apples left
This reasoning is incorrect because it adds the number of apples given away instead of subtracting them.

Now, solve this new problem using correct reasoning:
Problem: Tom has 10 candies and eats 4 of them. How many candies does Tom have left?

5. Present this prompt to the LLM and analyze its response to ensure it follows the correct reasoning pattern.

By using this Contrastive CoT Prompting technique, you provide the LLM with examples of both correct and incorrect reasoning, which helps it to better understand the problem-solving process and avoid common mistakes.
    
## Code Example






```python
import outlines
from outlines.integrations.utils import convert_json_schema_to_str
from pydantic import BaseModel, Field
from typing import List

model = outlines.models.transformers("google/gemma-2b")

class ReasoningStep(BaseModel):
    step: str = Field(..., description="A single step in the reasoning process")

class ArithmeticSolution(BaseModel):
    reasoning: List[ReasoningStep] = Field(..., description="List of reasoning steps")
    answer: int = Field(..., description="The final numerical answer")

schema_str = convert_json_schema_to_str(ArithmeticSolution.schema())

prompt = f"""Here's how to solve arithmetic word problems:

Correct example:
Problem: If Sarah has 5 apples and gives 2 to her friend, how many apples does she have left?
Reasoning: 
- Sarah starts with 5 apples
- She gives away 2 apples
- To find the remaining apples, we subtract: 5 - 2 = 3
- Therefore, Sarah has 3 apples left
This reasoning is correct because we subtract the number of apples given away from the initial number.

Incorrect example:
Problem: If Sarah has 5 apples and gives 2 to her friend, how many apples does she have left?
Reasoning:
- Sarah starts with 5 apples
- She gives away 2 apples
- To find the remaining apples, we add: 5 + 2 = 7
- Therefore, Sarah has 7 apples left
This reasoning is incorrect because it adds the number of apples given away instead of subtracting them.

Now, solve this new problem using correct reasoning:
Problem: Tom has 10 candies and eats 4 of them. How many candies does Tom have left?

Provide your reasoning steps and final answer in the following JSON schema: {schema_str}

Write down your reasoning process, step by step, before providing the final answer.
"""

generator = outlines.generate.json(model, ArithmeticSolution)
result = generator(prompt)
print(result)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    reasoning=[ReasoningStep(step='x = 10 - 4 = 6')] answer=6


