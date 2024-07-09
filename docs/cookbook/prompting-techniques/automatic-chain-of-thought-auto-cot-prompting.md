# Automatic Chain-of-Thought (Auto-CoT) Prompting


Auto-CoT is a technique that automatically generates chains of thought to build a few-shot prompt for complex reasoning tasks. It uses a zero-shot prompt to generate initial chains of thought for a set of training examples. These automatically generated chains are then used to construct a few-shot prompt that can be applied to new test samples. This approach aims to reduce the need for manual annotation of chain-of-thought reasoning while still benefiting from the improved performance of few-shot chain-of-thought prompting.
    

## A worked example


To implement Auto-CoT:

1. Start with a set of training examples for your task (e.g., math word problems).

2. Use a zero-shot CoT prompt like "Let's approach this step-by-step:" for each training example.

3. Generate chains of thought for each example using the zero-shot prompt.

4. Select a subset of these automatically generated examples with their chains of thought.

5. Construct a few-shot prompt by combining these examples.

6. Use this few-shot prompt for inference on new test samples.

For instance:

Training examples:
1. "If John has 5 apples and gives 2 to Mary, how many does he have left?"
2. "A train travels 120 miles in 2 hours. What is its average speed?"

Generate CoT for each (using GPT-3 or similar):
1. "Let's approach this step-by-step:
   1) John starts with 5 apples
   2) He gives 2 apples to Mary
   3) To find how many he has left, we subtract: 5 - 2 = 3
   Therefore, John has 3 apples left."

2. "Let's approach this step-by-step:
   1) The train travels 120 miles
   2) It takes 2 hours
   3) Speed is distance divided by time
   4) 120 miles / 2 hours = 60 miles per hour
   Therefore, the train's average speed is 60 mph."

Construct few-shot prompt:
"Solve the following problem step-by-step:

Q: If John has 5 apples and gives 2 to Mary, how many does he have left?
A: Let's approach this step-by-step:
1) John starts with 5 apples
2) He gives 2 apples to Mary
3) To find how many he has left, we subtract: 5 - 2 = 3
Therefore, John has 3 apples left.

Q: A train travels 120 miles in 2 hours. What is its average speed?
A: Let's approach this step-by-step:
1) The train travels 120 miles
2) It takes 2 hours
3) Speed is distance divided by time
4) 120 miles / 2 hours = 60 miles per hour
Therefore, the train's average speed is 60 mph.

Q: [New problem goes here]
A: Let's approach this step-by-step:"

Use this prompt for new problems, allowing the model to generate a chain of thought for the solution.
    
## Code Example


```python
from pydantic import BaseModel, Field
from typing import List
import outlines

class MathProblem(BaseModel):
    question: str
    steps: List[str] = Field(..., min_items=1)
    solution: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

# Generate initial chains of thought
cot_prompt = """
Solve the following math problem step-by-step:

Q: {question}
A: Let's approach this step-by-step:
"""

generator = outlines.generate.text(model)

training_problems = [
    "If John has 5 apples and gives 2 to Mary, how many does he have left?",
    "A train travels 120 miles in 2 hours. What is its average speed?"
]

cot_examples = []
for problem in training_problems:
    cot = generator(cot_prompt.format(question=problem), max_tokens=200)
    cot_examples.append(cot)

# Construct few-shot prompt
few_shot_prompt = "Solve the following math problems step-by-step:\n\n"
for problem, solution in zip(training_problems, cot_examples):
    few_shot_prompt += f"Q: {problem}\nA: {solution}\n\n"
few_shot_prompt += "Q: {question}\nA: Let's approach this step-by-step:"

# Function to solve new problems
def solve_problem(question: str) -> MathProblem:
    solution = generator(few_shot_prompt.format(question=question), max_tokens=300)
    lines = solution.strip().split('\n')
    steps = [line.strip('1234567890) ') for line in lines if line.strip().startswith(tuple('1234567890'))]
    final_solution = lines[-1] if lines[-1].startswith("Therefore") else "Unable to determine final solution."
    
    return MathProblem(question=question, steps=steps, solution=final_solution)

# Example usage
new_problem = "If a rectangle has a length of 10 cm and a width of 5 cm, what is its area?"
result = solve_problem(new_problem)
print(result)
```
    

