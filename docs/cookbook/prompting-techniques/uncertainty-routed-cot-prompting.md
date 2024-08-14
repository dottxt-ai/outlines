---
title: Uncertainty-Routed CoT Prompting
---

# Uncertainty-Routed CoT Prompting


Uncertainty-Routed CoT Prompting is an advanced technique that enhances the Chain-of-Thought (CoT) approach. It works by generating multiple reasoning paths for a given problem and then selecting the best answer based on a confidence threshold. Here's how it works:

1. The system generates multiple CoT reasoning paths for a given problem.
2. It then compares these paths to determine if there's a clear majority answer.
3. If the majority exceeds a predetermined confidence threshold, that answer is selected.
4. If there's no clear majority or the confidence is low, the system falls back to a greedy sampling approach, selecting the single most likely answer.

This technique is particularly useful for complex reasoning tasks where multiple valid thought processes might exist. By considering multiple paths and using a confidence-based selection method, it can often produce more reliable and accurate results than standard CoT approaches.
    

## A worked example


Let's walk through a simple example of how Uncertainty-Routed CoT Prompting might work for a basic math problem:

Problem: "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

Step 1: Generate multiple CoT reasoning paths (let's say we generate 3):

Path 1:
1. We need to find the average speed.
2. Speed is distance divided by time.
3. Distance is 120 miles.
4. Time is 2 hours.
5. 120 miles ÷ 2 hours = 60 miles per hour.

Path 2:
1. Average speed is total distance over total time.
2. We traveled 120 miles in 2 hours.
3. 120 ÷ 2 = 60.
4. The answer is 60 miles per hour.

Path 3:
1. Let's use the formula: Speed = Distance / Time
2. Distance = 120 miles
3. Time = 2 hours
4. Speed = 120 / 2 = 60 mph

Step 2: Compare the paths and check for a majority.
In this case, all three paths arrived at the same answer: 60 miles per hour.

Step 3: Check if the majority exceeds the confidence threshold.
Assuming our threshold is 66% agreement, we have 100% agreement here, so we exceed the threshold.

Step 4: Select the majority answer.
The system would confidently select 60 miles per hour as the final answer.

In this simple example, all paths led to the same conclusion. In more complex problems, you might see diverging paths, and the system would then either select a majority answer if it exceeds the threshold or fall back to greedy sampling if there's no clear consensus.

Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## Code Example





```python
import outlines
from pydantic import BaseModel, Field
from typing import List
from collections import Counter

model = outlines.models.transformers("google/gemma-2b")

class ReasoningStep(BaseModel):
    step: str = Field(..., description="A single step in the reasoning process")
    result: float = Field(..., description="Intermediate or final result of this step")

class MathProblem(BaseModel):
    problem: str = Field(..., description="The math problem to solve")
    steps: List[ReasoningStep] = Field(..., description="Steps in the reasoning process")
    final_answer: float = Field(..., description="The final answer to the problem")

def generate_multiple_reasoning_paths(prompt: str, num_attempts: int = 3):
    generator = outlines.generate.json(model, MathProblem)
    return [generator(prompt) for _ in range(num_attempts)]

def check_majority(answers: List[float], threshold: float = 0.66):
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    if most_common[1] / len(answers) >= threshold:
        return most_common[0]
    return None

prompt = """
Solve the following math problem step by step:
If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?

Provide your reasoning steps and the final answer.
"""

reasoning_attempts = generate_multiple_reasoning_paths(prompt)
final_answers = [attempt.final_answer for attempt in reasoning_attempts]

majority_answer = check_majority(final_answers)
if majority_answer is not None:
    final_result = majority_answer
else:
    final_result = reasoning_attempts[0].final_answer  # Fallback to greedy selection

print(f"Final answer: {final_result} miles per hour")
print("\nReasoning paths:")
for i, attempt in enumerate(reasoning_attempts, 1):
    print(f"\nAttempt {i}:")
    for step in attempt.steps:
        print(f"- {step.step} (Result: {step.result})")
    print(f"Final answer: {attempt.final_answer}")
```

    `config.hidden_act` is ignored, you should use `config.hidden_activation` instead.
    Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use
    `config.hidden_activation` if you want to override this behaviour.
    See https://github.com/huggingface/transformers/pull/29402 for more details.



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|███████████████| 113/113 [00:04<00:00, 28.04it/s]


    Final answer: 60.0 miles per hour
    
    Reasoning paths:
    
    Attempt 1:
    - a = 120, b = 2, c: 2, x: 0, y: 120: (Result: 60.0)
    - x = bc, x: 60 (Result: 60.0)
    - y = ac, y: 80 (Result: 80.0)
    - v = x/y, v = 60/80 = 0.75 (Result: 0.75)
    Final answer: 0.75
    
    Attempt 2:
    - Step 1 (Result: 120.0)
    Final answer: 60.0
    
    Attempt 3:
    - A 120 mile train could go 240 miles in 2 hours (Result: 60.0)
    - The train goes 60 miles in every 1 hour (Result: 60.0)
    - Answer (Result: 60.0)
    Final answer: 60.0



```python

```
