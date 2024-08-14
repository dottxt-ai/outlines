---
title: Chain-of-Thought (CoT) Prompting
---

# Chain-of-Thought (CoT) Prompting


Chain-of-Thought (CoT) Prompting is a technique that encourages large language models to express their reasoning process before providing a final answer. It typically uses few-shot prompting, where example questions with their corresponding thought processes and answers are provided. This approach has been shown to significantly improve performance on tasks requiring complex reasoning, such as mathematics problems.

The key idea is to guide the model to break down its thinking into smaller, logical steps, mimicking human problem-solving. By doing so, the model can tackle more complex problems and provide more accurate answers, as it's essentially "showing its work."
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


Here's how to implement Chain-of-Thought Prompting:

1. Choose a task that requires reasoning (e.g., a math word problem).

2. Create a prompt that includes:
   a) An example question
   b) A step-by-step reasoning process
   c) The correct answer

3. Add your actual question after the example.

Here's a simple implementation:

Prompt:

Q: Sarah has 3 apples. She buys 2 more apples and then gives 1 apple to her friend. How many apples does Sarah have now?
A: Let's think through this step-by-step:
1. Sarah starts with 3 apples
2. She buys 2 more apples, so now she has 3 + 2 = 5 apples
3. She gives 1 apple to her friend, so we subtract 1 from 5
4. 5 - 1 = 4 apples
Therefore, Sarah now has 4 apples.

Q: Tom has $50. He spends $30 on a book and then receives $15 as a gift. How much money does Tom have now?
A: Let's think through this step-by-step:

4. Send this prompt to the language model.

5. The model should respond with a step-by-step reasoning process followed by the final answer, like this:


1. Tom starts with $50
2. He spends $30 on a book, so now he has $50 - $30 = $20
3. He receives $15 as a gift, so we add $15 to $20
4. $20 + $15 = $35
Therefore, Tom now has $35.

By using this technique, you encourage the model to break down the problem and show its reasoning, which often leads to more accurate results.
    
## Code Example






```python
from pydantic import BaseModel, Field

class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="Reasoning step")

from typing import List

class Reasoning(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: str = Field(..., description="Conclusion")

json_schema = Reasoning.model_json_schema()

from outlines.integrations.utils import convert_json_schema_to_str

schema_str = convert_json_schema_to_str(json_schema=json_schema)
```


```python
import outlines

model = outlines.models.transformers("google/gemma-2b")

prompt = f"""Use chain-of-thought reasoning to solve the following problem.

Your answer should be in the following JSON format:
{schema_str}

Now, solve this problem:

Q: Tom has $50. He spends $30 on a book and then receives $15 as a gift. How much money does Tom have now?
A:
"""

generator = outlines.generate.json(model, Reasoning)
result = generator(prompt)

print(result.reasoning)
print(f"Final answer: {result.conclusion}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    [Reasoning_Step(reasoning_step='Bought $30 book'), Reasoning_Step(reasoning_step='Received $15 gift')]
    Final answer: Tom has $45


We might want to consider providing some examples to improve this result.


```python

```
