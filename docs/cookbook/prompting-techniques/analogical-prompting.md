# Analogical Prompting


Analogical Prompting automatically generates exemplars that include chains-of-thought (CoT) reasoning for a given task. It aims to improve performance on complex reasoning tasks by providing the language model with relevant analogies that demonstrate the problem-solving process. This technique leverages the model's ability to draw parallels between similar problems and apply analogous reasoning strategies.
    

## A worked example


1. Define the target task (e.g. solving a complex math word problem)

2. Generate a pool of similar but simpler problems that use analogous reasoning:
   - Use the language model to generate variations on the target problem
   - Retrieve relevant problems from a curated dataset

3. For each problem in the pool, generate a chain-of-thought solution:
   - Prompt the model to solve the problem step-by-step
   - Review and refine the generated solutions

4. Select the best analogous problems and solutions to use as exemplars:
   - Choose examples that demonstrate key reasoning steps
   - Ensure diversity in the selected analogies

5. Construct the final prompt by combining:
   - The target problem
   - 2-3 selected analogous problems with their CoT solutions
   - An instruction to solve the target problem using similar reasoning

6. Submit the constructed prompt to the language model to solve the target problem

Example prompt:
"Here are some example math word problems with solutions:

Problem 1: Tom has 3 apples and buys 2 more. How many does he have?
Solution: Let's approach this step-by-step:
1. Tom starts with 3 apples
2. He buys 2 more apples
3. To find the total, we add: 3 + 2 = 5
So Tom has 5 apples in total.

Problem 2: A recipe needs 2 cups of flour for 4 servings. How much flour for 6 servings?
Solution: We can solve this with these steps:
1. The recipe uses 2 cups for 4 servings
2. For 6 servings, we need to scale up
3. 6 servings is 1.5 times 4 servings (6 ÷ 4 = 1.5)
4. So we need 1.5 times the flour: 2 * 1.5 = 3
Therefore, we need 3 cups of flour for 6 servings.

Now solve this problem using similar reasoning:
Sarah bakes cookies that require 1.5 cups of sugar to make 3 dozen. How much sugar does she need to make 5 dozen cookies?"
    
## Code Example


```python
from pydantic import BaseModel, Field
import outlines

class MathProblem(BaseModel):
    problem: str
    solution: str = Field(..., description="Step-by-step solution to the problem")

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, MathProblem)

def generate_analogous_problem(topic: str) -> MathProblem:
    prompt = f"Generate a simple {topic} math problem with a step-by-step solution."
    return generator(prompt)

def solve_problem_with_analogies(target_problem: str, num_analogies: int = 2) -> str:
    # Generate analogous problems
    analogies = [generate_analogous_problem("arithmetic") for _ in range(num_analogies)]
    
    # Construct the prompt with analogies
    prompt = "Here are some example math word problems with solutions:\n\n"
    for i, analogy in enumerate(analogies, 1):
        prompt += f"Problem {i}: {analogy.problem}\n"
        prompt += f"Solution: {analogy.solution}\n\n"
    
    prompt += f"Now solve this problem using similar reasoning:\n{target_problem}"
    
    # Generate the solution for the target problem
    solution = generator(prompt)
    return solution.solution

# Example usage
target_problem = "Sarah bakes cookies that require 1.5 cups of sugar to make 3 dozen. How much sugar does she need to make 5 dozen cookies?"
result = solve_problem_with_analogies(target_problem)
print(result)
```
    

