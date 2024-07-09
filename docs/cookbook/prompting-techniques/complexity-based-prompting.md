# Complexity-based Prompting


Complexity-based Prompting is an advanced chain-of-thought (CoT) technique that involves two key modifications:

1. Example Selection: It selects complex examples for annotation and inclusion in the prompt, based on factors like question length or number of reasoning steps required.

2. Inference Strategy: During inference, it samples multiple reasoning chains (answers) and uses a majority vote among chains exceeding a certain length threshold. This is based on the premise that longer reasoning indicates higher answer quality.

This technique aims to improve the language model's performance on complex reasoning tasks by exposing it to more challenging examples and favoring more detailed responses.
    

## A worked example


To implement Complexity-based Prompting:

1. Prepare a dataset of problems in your target domain (e.g., mathematical reasoning).

2. Analyze the problems to determine their complexity:
   - Count the number of words or tokens in each question
   - Estimate the number of reasoning steps required
   - Use other relevant complexity metrics for your domain

3. Select the most complex examples (e.g., top 20%) for annotation.

4. Create detailed chain-of-thought annotations for these complex examples, showing step-by-step reasoning.

5. Construct your prompt by including these annotated complex examples.

6. For inference:
   a. Generate multiple (e.g., 5-10) reasoning chains for the input question.
   b. Measure the length of each reasoning chain.
   c. Set a length threshold (e.g., 75th percentile of chain lengths).
   d. Keep only the chains that exceed this threshold.
   e. Take a majority vote among the remaining chains to determine the final answer.

7. If no clear majority emerges, you may choose to fall back to a simpler method (e.g., selecting the longest chain) or indicate low confidence in the result.

Example prompt structure:
```
Here are some complex mathematical reasoning problems with step-by-step solutions:

Problem 1: [Complex question]
Solution:
Step 1: [Reasoning]
Step 2: [Reasoning]
...
Final Answer: [Answer]

[Include 2-3 more complex examples]

Now, solve this problem step-by-step:
[New problem to solve]
```

Repeat the inference process multiple times, then apply the length threshold and majority voting to determine the final answer.
    
## Code Example


```python
from pydantic import BaseModel, Field
from typing import List
import outlines
from statistics import mean

class ReasoningStep(BaseModel):
    step_number: int
    description: str

class Solution(BaseModel):
    steps: List[ReasoningStep]
    final_answer: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, Solution)

prompt = """Here are some complex mathematical reasoning problems with step-by-step solutions:

Problem 1: If a train travels 120 km in 2 hours, what is its average speed in km/h?
Solution:
Step 1: Understand that average speed is calculated by dividing distance by time.
Step 2: The distance traveled is 120 km.
Step 3: The time taken is 2 hours.
Step 4: Divide 120 km by 2 hours.
Final Answer: The average speed is 60 km/h.

Problem 2: A rectangle has a length that is 3 times its width. If the perimeter of the rectangle is 48 cm, what are its dimensions?
Solution:
Step 1: Let the width be x. Then the length is 3x.
Step 2: The perimeter formula for a rectangle is 2(length + width).
Step 3: Substitute the values: 48 = 2(3x + x)
Step 4: Simplify: 48 = 2(4x) = 8x
Step 5: Solve for x: x = 48 / 8 = 6
Step 6: Calculate the length: 3x = 3 * 6 = 18
Final Answer: The width is 6 cm and the length is 18 cm.

Now, solve this problem step-by-step:
A car travels 240 miles in 4 hours. If it continues at the same speed, how long will it take to travel an additional 180 miles?
"""

# Generate multiple reasoning chains
num_chains = 5
solutions = [generator(prompt) for _ in range(num_chains)]

# Calculate the average number of steps
avg_steps = mean(len(solution.steps) for solution in solutions)

# Set a threshold for chain length (e.g., 75% of the average)
threshold = 0.75 * avg_steps

# Filter solutions that meet the threshold
valid_solutions = [sol for sol in solutions if len(sol.steps) >= threshold]

# Majority vote for the final answer
if valid_solutions:
    final_answers = [sol.final_answer for sol in valid_solutions]
    majority_answer = max(set(final_answers), key=final_answers.count)
    print(f"Majority Answer: {majority_answer}")
else:
    print("No clear majority. Consider using the longest chain or indicate low confidence.")
```
    

