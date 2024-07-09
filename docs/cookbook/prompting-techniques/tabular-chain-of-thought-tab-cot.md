# Tabular Chain-of-Thought (Tab-CoT)


Tab-CoT is a zero-shot prompting technique that encourages large language models to output their reasoning in a structured, tabular format using markdown. This approach aims to improve the model's ability to break down complex problems into smaller, manageable steps while maintaining a clear and organized thought process. The technique involves crafting a prompt that instructs the model to present its reasoning as a markdown table, with each row representing a step in the problem-solving process.
    

## A worked example


To implement the Tab-CoT technique:

1. Start with your question or problem statement.

2. Craft a prompt that includes:
   a) The original question
   b) Instructions to use a markdown table for reasoning
   c) A thought-inducing phrase

3. Format the prompt as follows:

```
Question: [Insert your question here]

Please solve this problem step-by-step using a markdown table. Each row should represent a step in your reasoning process.

| Step | Reasoning |
|------|-----------|
| 1    | [First step goes here] |
| 2    | [Second step goes here] |
...

Let's approach this systematically:
```

4. Send this prompt to the language model.

5. The model should respond with a markdown table containing its step-by-step reasoning, followed by a final answer.

Example implementation:

Question: "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

Prompt:
```
Question: If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?

Please solve this problem step-by-step using a markdown table. Each row should represent a step in your reasoning process.

| Step | Reasoning |
|------|-----------|
| 1    | [First step goes here] |
| 2    | [Second step goes here] |
...

Let's approach this systematically:
```

Expected model output:

```
| Step | Reasoning |
|------|-----------|
| 1    | Understand the given information: Distance = 120 miles, Time = 2 hours |
| 2    | Recall the formula for average speed: Average Speed = Distance ÷ Time |
| 3    | Plug in the values: Average Speed = 120 miles ÷ 2 hours |
| 4    | Perform the calculation: 120 ÷ 2 = 60 |
| 5    | Check the units: The result is in miles per hour (mph) |

The average speed of the train is 60 miles per hour (mph).
```

This technique helps the model break down its reasoning into clear, discrete steps, potentially improving its problem-solving capabilities and making its thought process more transparent to users.
    
## Code Example


```python
import outlines
from pydantic import BaseModel
from typing import List, Tuple

class TabCoTResult(BaseModel):
    reasoning_table: List[Tuple[int, str]]
    final_answer: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.format(model, str)

prompt = """Question: If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?

Please solve this problem step-by-step using a markdown table. Each row should represent a step in your reasoning process.

| Step | Reasoning |
|------|-----------|
| 1    | [First step goes here] |
| 2    | [Second step goes here] |
...

Let's approach this systematically:
"""

response = generator(prompt)

# Parse the response
lines = response.split('\n')
table_start = lines.index('| Step | Reasoning |') + 2
table_end = lines.index('', table_start)
table_rows = [line.split('|')[1:3] for line in lines[table_start:table_end]]
reasoning_table = [(int(step.strip()), reasoning.strip()) for step, reasoning in table_rows]

final_answer = lines[table_end + 1].strip()

result = TabCoTResult(reasoning_table=reasoning_table, final_answer=final_answer)
print(result)
```
    

