# Active Prompting


Active Prompting is an iterative technique that aims to improve few-shot prompts by identifying and refining challenging exemplars. It starts with a set of training questions/exemplars, has the language model solve them, then calculates uncertainty (measured by disagreement among multiple runs) for each exemplar. Human annotators are then asked to rewrite the exemplars with the highest uncertainty. This process creates more effective few-shot prompts by focusing human effort on the most difficult cases.
    

## A worked example


1. Start with a set of 20 training questions/exemplars for a math reasoning task.

2. Use these exemplars in a few-shot Chain-of-Thought (CoT) prompt and have the language model solve them 5 times each.

3. Calculate uncertainty for each exemplar by measuring disagreement among the 5 runs. For example, if an exemplar gets 3 different answers across 5 runs, it has high uncertainty.

4. Rank the exemplars by uncertainty and select the top 5 most uncertain ones.

5. Have human annotators rewrite these 5 exemplars to make them clearer or more informative. For instance, they might add more detailed reasoning steps or clarify ambiguous wording.

6. Replace the original 5 uncertain exemplars with the rewritten versions in your prompt.

7. Use this refined prompt for your actual task, potentially leading to improved performance due to the more effective exemplars.

8. Optionally, repeat steps 2-7 for multiple iterations to further refine the prompt.
    
## Code Example


```python
from pydantic import BaseModel, Field
from typing import List
import outlines
from enum import Enum
import random

class MathProblem(BaseModel):
    question: str
    solution: str
    uncertainty: float = Field(default=0.0)

class Uncertainty(float, Enum):
    LOW = 0.0
    MEDIUM = 0.5
    HIGH = 1.0

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

def generate_solution(problem: MathProblem) -> str:
    prompt = f"Solve this math problem: {problem.question}\nSolution:"
    generator = outlines.generate.text(model)
    return generator(prompt, max_tokens=100)

def calculate_uncertainty(solutions: List[str]) -> float:
    unique_solutions = set(solutions)
    if len(unique_solutions) == 1:
        return Uncertainty.LOW
    elif len(unique_solutions) == len(solutions):
        return Uncertainty.HIGH
    else:
        return Uncertainty.MEDIUM

def active_prompting(problems: List[MathProblem], iterations: int = 1) -> List[MathProblem]:
    for _ in range(iterations):
        for problem in problems:
            solutions = [generate_solution(problem) for _ in range(5)]
            problem.uncertainty = calculate_uncertainty(solutions)
        
        problems.sort(key=lambda x: x.uncertainty, reverse=True)
        top_uncertain = problems[:5]
        
        # Simulated human refinement (in reality, this would involve human input)
        for problem in top_uncertain:
            problem.question += " Please provide a step-by-step solution."
            problem.uncertainty = Uncertainty.LOW
    
    return problems

# Example usage
initial_problems = [
    MathProblem(question="What is 5 + 7?", solution=""),
    MathProblem(question="Solve for x: 2x + 3 = 11", solution=""),
    MathProblem(question="Calculate the area of a circle with radius 4", solution=""),
]

refined_problems = active_prompting(initial_problems, iterations=2)
for problem in refined_problems:
    print(f"Question: {problem.question}")
    print(f"Uncertainty: {problem.uncertainty}")
    print("---")
```
    

