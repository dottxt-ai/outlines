# Uncertainty-Routed CoT Prompting


Uncertainty-Routed CoT Prompting is an advanced chain-of-thought technique that aims to improve the reliability of LLM responses. It works by generating multiple reasoning paths for a given problem, then selecting the final answer based on the consistency and confidence of those paths. The key steps are:

1. Generate multiple CoT reasoning paths for the input question
2. Analyze the consistency of the final answers across paths
3. If there is high consistency (above a predetermined threshold), select the majority answer
4. If consistency is low, fall back to a single greedy sampling path

This technique leverages the insight that when an LLM is more certain about an answer, it tends to produce consistent results across multiple samplings. By routing uncertain cases to a more conservative approach, it aims to reduce errors on difficult problems.
    

## A worked example


Here's a step-by-step example of implementing Uncertainty-Routed CoT Prompting:

1. Input question: 
   "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

2. Generate multiple CoT paths (e.g., 5 paths):
   Path 1: "Let's approach this step-by-step:
   1) We know distance = 120 miles
   2) We know time = 2 hours
   3) Speed is calculated as distance divided by time
   4) So, speed = 120 miles / 2 hours
   5) 120 / 2 = 60
   Therefore, the average speed is 60 miles per hour."

   Path 2-5: [Similar reasoning, all concluding 60 mph]

3. Analyze consistency:
   - All 5 paths concluded 60 mph
   - Consistency is 100% (5/5 agree)

4. Decision:
   - Consistency exceeds threshold (e.g., 80%)
   - Select majority answer: 60 mph

5. Output final answer: "The train's average speed is 60 miles per hour."

If the paths had produced inconsistent results (e.g., some saying 60 mph, others 30 mph), and consistency fell below the threshold, the system would fall back to using a single greedy sampling path instead.
    
## Code Example


```python
from pydantic import BaseModel
from typing import List
import outlines
from collections import Counter

class UncertaintyRoutedCoT(BaseModel):
    question: str
    cot_paths: List[str]
    final_answer: float
    consistency: float

model = outlines.models.transformers("WizardLM/WizardMath-7B-V1.1")

def generate_cot_path(question: str) -> str:
    prompt = f"""Solve this math problem step by step:
{question}
Show your reasoning and final answer."""
    return outlines.generate.text(model)(prompt, max_tokens=200)

def extract_answer(cot_path: str) -> float:
    prompt = f"""Extract the final numerical answer from this reasoning:
{cot_path}
Final answer:"""
    return outlines.generate.format(model, float)(prompt)

def uncertainty_routed_cot(question: str, num_paths: int = 5, consistency_threshold: float = 0.8) -> UncertaintyRoutedCoT:
    cot_paths = [generate_cot_path(question) for _ in range(num_paths)]
    answers = [extract_answer(path) for path in cot_paths]
    
    answer_counts = Counter(answers)
    most_common_answer, count = answer_counts.most_common(1)[0]
    consistency = count / num_paths
    
    if consistency >= consistency_threshold:
        final_answer = most_common_answer
    else:
        final_answer = extract_answer(generate_cot_path(question))
    
    return UncertaintyRoutedCoT(
        question=question,
        cot_paths=cot_paths,
        final_answer=final_answer,
        consistency=consistency
    )

question = "If a train travels 180 miles in 3 hours, what is its average speed in miles per hour?"
result = uncertainty_routed_cot(question)
print(result)
```
    

