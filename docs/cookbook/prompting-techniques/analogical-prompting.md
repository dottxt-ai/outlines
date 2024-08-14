# Analogical Prompting


Analogical Prompting is an advanced prompting technique that automatically generates exemplars including Chain-of-Thought (CoT) reasoning. It works by creating an analogous problem to the target problem, demonstrating the step-by-step reasoning process for solving that analogous problem, and then presenting the target problem. This allows the language model to apply similar reasoning to solve the new problem.
    

## A worked example


Here's how to implement Analogical Prompting for a simple math word problem:

1. Define the target problem:
   "Sarah has 3 bags of marbles. Each bag contains 12 marbles. How many marbles does Sarah have in total?"

2. Generate an analogous problem:
   "John has 4 boxes of chocolates. Each box contains 6 chocolates. How many chocolates does John have in total?"

3. Provide CoT reasoning for the analogous problem:
   "Let's solve this step by step:
   1. John has 4 boxes of chocolates
   2. Each box contains 6 chocolates
   3. To find the total, we multiply: 4 x 6 = 24
   Therefore, John has 24 chocolates in total."

4. Present the full prompt:

   "Problem: John has 4 boxes of chocolates. Each box contains 6 chocolates. How many chocolates does John have in total?

   Solution:
   Let's solve this step by step:
   1. John has 4 boxes of chocolates
   2. Each box contains 6 chocolates
   3. To find the total, we multiply: 4 x 6 = 24
   Therefore, John has 24 chocolates in total.

   Now, solve this problem:
   Sarah has 3 bags of marbles. Each bag contains 12 marbles. How many marbles does Sarah have in total?"

5. The language model should now apply analogous reasoning to solve the target problem.
    
## Code Example






```python
import outlines
from pydantic import BaseModel, Field
from typing import List

# Define the model for math word problems
class MathProblem(BaseModel):
    subject: str = Field(..., description="The subject of the problem")
    item: str = Field(..., description="The item being counted")
    containers: int = Field(..., description="Number of containers")
    items_per_container: int = Field(..., description="Number of items in each container")

# Define the model for solution steps
class Solution(BaseModel):
    steps: List[str] = Field(..., description="Step-by-step solution")
    result: int = Field(..., description="Final result")

# Initialize the language model
model = outlines.models.transformers("google/gemma-2b")

```

    `config.hidden_act` is ignored, you should use `config.hidden_activation` instead.
    Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use
    `config.hidden_activation` if you want to override this behaviour.
    See https://github.com/huggingface/transformers/pull/29402 for more details.



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]



```python

from outlines.integrations.utils import convert_json_schema_to_str
schema_str = convert_json_schema_to_str(json_schema=Solution.model_json_schema())
# Create the Analogical Prompting function
def analogical_prompt(problem: MathProblem) -> str:
    return f"""
Analogous Problem: John has 4 boxes of chocolates. Each box contains 6 chocolates. How many chocolates does John have in total?

Analogous Solution:
Let's solve this step by step:
1. John has 4 boxes of chocolates
2. Each box contains 6 chocolates
3. To find the total, we multiply: 4 x 6 = 24
Therefore, John has 24 chocolates in total.

Now, solve this problem: {problem.subject} has {problem.containers} {problem.item}s of {problem.item}s. Each {problem.item} contains {problem.items_per_container} {problem.item}s. How many {problem.item}s does {problem.subject} have in total?

Provide a step-by-step solution and the final result in JSON format according to this schema: {schema_str}
    """

# Generate the solution using Analogical Prompting
problem = MathProblem(subject="Sarah", item="bag", containers=3, items_per_container=12)
```


```python
prompt = analogical_prompt(problem)
print(prompt)
```

    
    Analogous Problem: John has 4 boxes of chocolates. Each box contains 6 chocolates. How many chocolates does John have in total?
    
    Analogous Solution:
    Let's solve this step by step:
    1. John has 4 boxes of chocolates
    2. Each box contains 6 chocolates
    3. To find the total, we multiply: 4 x 6 = 24
    Therefore, John has 24 chocolates in total.
    
    Now, solve this problem: Sarah has 3 bags of bags. Each bag contains 12 bags. How many bags does Sarah have in total?
    
    Provide a step-by-step solution and the final result in JSON format according to this schema: {"properties": {"steps": {"description": "Step-by-step solution", "items": {"type": "string"}, "title": "Steps", "type": "array"}, "result": {"description": "Final result", "title": "Result", "type": "integer"}}, "required": ["steps", "result"], "title": "Solution", "type": "object"}
        



```python
generator = outlines.generate.json(model, Solution)
solution = generator(prompt)

print(f"Problem: {problem}")
print(f"Solution: {solution}")
```

    Problem: subject='Sarah' item='bag' containers=3 items_per_container=12
    Solution: steps=['Step 1: Divide the given values', '$({2} x {2}) - {1} = {0}$'] result=3


We can see here that the small model struggles to get the right answer. We might want to explore providing more examples.

