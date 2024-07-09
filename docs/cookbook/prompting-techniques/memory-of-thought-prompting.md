# Memory-of-Thought Prompting


Memory-of-Thought Prompting leverages unlabeled training examples to build effective Few-Shot Chain-of-Thought (CoT) prompts at test time. The technique works in two main phases:

1. Pre-processing: Before test time, the system performs inference on a set of unlabeled training examples using a Chain-of-Thought approach. This generates reasoning chains for each example.

2. Test-time retrieval: When given a new test sample, the system retrieves similar instances from the pre-processed training set. It then uses these similar examples and their generated reasoning chains to construct a Few-Shot CoT prompt tailored to the specific test sample.

This approach allows the model to dynamically create relevant Few-Shot prompts for each test instance, improving performance on tasks like arithmetic, commonsense, and factual reasoning.
    

## A worked example


Here's a step-by-step implementation of Memory-of-Thought Prompting:

1. Collect a set of unlabeled training examples relevant to your task domain (e.g., arithmetic word problems).

2. Pre-processing:
   a. For each training example, use a Zero-Shot CoT prompt to generate a reasoning chain.
   Example prompt: "Solve this arithmetic problem step by step: [PROBLEM]"
   b. Store each problem along with its generated reasoning chain.

3. Implement a similarity function to compare problems (e.g., cosine similarity of sentence embeddings).

4. At test time:
   a. Given a new test problem, use the similarity function to retrieve the top-k most similar problems from the pre-processed set.
   b. Construct a Few-Shot CoT prompt using the retrieved problems and their reasoning chains.
   c. Append the test problem to this prompt.

5. Submit the constructed prompt to the language model to solve the test problem.

Example test-time prompt:
"""
Solve these arithmetic problems step by step:

Problem 1: John has 5 apples. He gives 2 to his friend and buys 3 more. How many apples does John have now?
Step 1: Start with John's initial apples: 5
Step 2: Subtract the apples he gave away: 5 - 2 = 3
Step 3: Add the apples he bought: 3 + 3 = 6
Therefore, John now has 6 apples.

Problem 2: [Another retrieved similar problem and its reasoning chain]

Problem 3: [Another retrieved similar problem and its reasoning chain]

Now solve this new problem:
[TEST PROBLEM]
"""

This dynamic prompt construction allows the model to leverage relevant examples and reasoning patterns for each specific test instance.
    
## Code Example


```python
from typing import List
from pydantic import BaseModel
import outlines
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Pydantic models for structured data
class ReasoningStep(BaseModel):
    step_number: int
    description: str

class ArithmeticProblem(BaseModel):
    problem: str
    reasoning: List[ReasoningStep]

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

# Pre-process examples (simulated here)
preprocessed_examples = [
    ArithmeticProblem(
        problem="John has 5 apples. He gives 2 to his friend and buys 3 more. How many apples does John have now?",
        reasoning=[
            ReasoningStep(step_number=1, description="Start with John's initial apples: 5"),
            ReasoningStep(step_number=2, description="Subtract the apples he gave away: 5 - 2 = 3"),
            ReasoningStep(step_number=3, description="Add the apples he bought: 3 + 3 = 6"),
        ]
    ),
    # Add more pre-processed examples here
]

def get_similar_problems(query: str, examples: List[ArithmeticProblem], top_k: int = 2) -> List[ArithmeticProblem]:
    query_embedding = embedding_model.encode([query])
    example_embeddings = embedding_model.encode([ex.problem for ex in examples])
    similarities = cosine_similarity(query_embedding, example_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [examples[i] for i in top_indices]

def construct_prompt(similar_problems: List[ArithmeticProblem], test_problem: str) -> str:
    prompt = "Solve these arithmetic problems step by step:\n\n"
    for idx, problem in enumerate(similar_problems, 1):
        prompt += f"Problem {idx}: {problem.problem}\n"
        for step in problem.reasoning:
            prompt += f"Step {step.step_number}: {step.description}\n"
        prompt += "\n"
    prompt += f"Now solve this new problem:\n{test_problem}\n"
    return prompt

# Example usage
test_problem = "If Sarah has 10 candies and eats 3, then receives 5 more from her mom, how many candies does she have?"
similar_problems = get_similar_problems(test_problem, preprocessed_examples)
prompt = construct_prompt(similar_problems, test_problem)

# Generate solution using outlines
generator = outlines.generate.text(llm)
solution = generator(prompt, max_tokens=200)
print(solution)
```
    

