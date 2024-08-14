# Active Prompting


Active Prompting is an iterative technique that involves dynamically refining prompts based on the model's responses. This method aims to improve the quality and relevance of the model's outputs by continuously adjusting the input. The process begins with an initial prompt, followed by an evaluation of the model's response. Based on this evaluation, the prompt is modified to address any shortcomings or to further guide the model towards the desired output. This cycle of prompting, evaluating, and refining continues until the desired quality or specificity of response is achieved.


## Step by Step Example


Let's say we want to get a concise explanation of photosynthesis. Here's how we might use Active Prompting:

1. Initial prompt: "Explain photosynthesis."
   Model response: [Long, detailed explanation with technical terms]

2. Refined prompt: "Explain photosynthesis in simple terms, suitable for a middle school student."
   Model response: [Simpler explanation, but still too long]

3. Further refined prompt: "Provide a brief, 2-3 sentence explanation of photosynthesis for a middle school student."
   Model response: [Concise explanation, but missing a key point about energy]

4. Final prompt: "In 2-3 sentences, explain photosynthesis for a middle school student, making sure to mention that it produces energy for the plant."
   Model response: [Satisfactory concise explanation including the energy aspect]

This process demonstrates how we actively refine the prompt based on each response, gradually steering the model towards producing the desired output.

## Code Example






```python
import outlines
from pydantic import BaseModel, Field
from typing import List, Union

# Define our Pydantic models
class MathProblem(BaseModel):
    operation: str
    numbers: List[int]
    difficulty: str = Field(..., description="easy, medium, or hard")

class MathSolution(BaseModel):
    problem: MathProblem
    solution: int
    explanation: str

@outlines.prompt
def math_problem_solver(problem: MathProblem):
    """
    Solve the following math problem:
    Operation: {problem.operation}
    Numbers: {problem.numbers}
    Difficulty: {problem.difficulty}

    Provide the solution and a brief explanation of how you solved it.
    """

# Initialize our model
model = outlines.models.transformers("google/gemma-2-2b", device='mps')

# Function to generate and solve problems
def solve_problem(problem: MathProblem) -> MathSolution:
    prompt = math_problem_solver(problem)
    return outlines.generate.json(model, MathSolution)(prompt)

# Active Prompting loop
def active_prompting_math():
    problems = [
        MathProblem(operation="addition", numbers=[5, 3], difficulty="easy"),
        MathProblem(operation="multiplication", numbers=[6, 7], difficulty="medium"),
        MathProblem(operation="exponentiation", numbers=[2, 3], difficulty="hard")
    ]
    
    for problem in problems:
        solution = solve_problem(problem)
        print(f"Problem: {solution.problem}")
        print(f"Solution: {solution.solution}")
        print(f"Explanation: {solution.explanation}")
        print("---")

# Run the Active Prompting process
active_prompting_math()
```


    Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|██████████| 136/136 [00:05<00:00, 23.61it/s]


    Problem: operation='addition' numbers=[1, 2, 3] difficulty='}, '
    Solution: 6
    Explanation: Adding all three numbers results in 6.
    ---


    Compiling FSM index for all state transitions: 100%|██████████| 136/136 [00:05<00:00, 23.35it/s]


    Problem: operation='-' numbers=[4, 6] difficulty=''
    Solution: -2
    Explanation: Simple multiplication of two numbers and then a simple subtraction. If the whole operation is done well, it will have a different answer from the actual picture. Please do it once or twice. 1 + (-4) = -3
    ---
    Problem: operation='div' numbers=[123, 87] difficulty='hard'
    Solution: 14
    Explanation: Divide 123 and 87 to get 14.
    ---



This example demonstrates Active Prompting by progressively increasing the difficulty and complexity of math problems. It uses Pydantic models to structure the input (MathProblem) and output (MathSolution), ensuring that we capture all necessary information for each iteration.

The `active_prompting_math` function simulates the iterative process by presenting a series of problems with increasing difficulty. In a more advanced implementation, you could add logic to dynamically adjust the difficulty based on the model's performance on previous problems.

This approach showcases how Active Prompting can be used to assess and challenge a model's capabilities, gradually moving from simple tasks to more complex ones based on its demonstrated abilities.




