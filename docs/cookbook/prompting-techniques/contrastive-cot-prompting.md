# Contrastive CoT Prompting


Contrastive CoT Prompting enhances Chain-of-Thought prompting by including both correct and incorrect reasoning examples in the prompt. This technique shows the language model both how to reason properly and how not to reason, helping it distinguish between valid and invalid logical steps. By contrasting correct and incorrect chains of thought, the model can better understand the nuances of proper reasoning and avoid common pitfalls.
    

## A worked example


To implement Contrastive CoT Prompting:

1. Select a task or problem type you want the model to solve (e.g., arithmetic word problems).

2. Create several example problems with both correct and incorrect reasoning chains:

Correct example:
Q: If John has 5 apples and gives 2 to his friend, how many apples does he have left?
A: Let's approach this step-by-step:
1. John starts with 5 apples
2. He gives away 2 apples
3. To find how many are left, we subtract: 5 - 2 = 3
Therefore, John has 3 apples left.

Incorrect example:
Q: If John has 5 apples and gives 2 to his friend, how many apples does he have left?
A: Let's think about this:
1. John starts with 5 apples
2. He gives away 2 apples
3. To find how many are left, we add: 5 + 2 = 7
This reasoning is incorrect because we should subtract, not add.

3. Combine these examples into a prompt, clearly labeling the correct and incorrect reasoning:

"Here are two examples of solving arithmetic word problems. The first shows correct reasoning, while the second shows incorrect reasoning:

[Insert correct example]

[Insert incorrect example]

Now, solve the following problem using correct reasoning:

[Insert new problem to solve]"

4. Use this prompt with your chosen language model to generate solutions for new problems. The model should now be primed to use correct reasoning patterns and avoid common mistakes.

5. Evaluate the model's performance and iterate on your examples if necessary, potentially adding more contrastive pairs to reinforce the correct reasoning process.
    
## Code Example


```python
from pydantic import BaseModel, conint
import outlines

class ArithmeticSolution(BaseModel):
    steps: list[str]
    answer: conint(gt=0)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, ArithmeticSolution)

prompt = """Here are two examples of solving arithmetic word problems. The first shows correct reasoning, while the second shows incorrect reasoning:

Correct example:
Q: If John has 5 apples and gives 2 to his friend, how many apples does he have left?
A: Let's approach this step-by-step:
1. John starts with 5 apples
2. He gives away 2 apples
3. To find how many are left, we subtract: 5 - 2 = 3
Therefore, John has 3 apples left.

Incorrect example:
Q: If John has 5 apples and gives 2 to his friend, how many apples does he have left?
A: Let's think about this:
1. John starts with 5 apples
2. He gives away 2 apples
3. To find how many are left, we add: 5 + 2 = 7
This reasoning is incorrect because we should subtract, not add.

Now, solve the following problem using correct reasoning:

Q: If Mary has 10 cookies and eats 3, how many cookies does she have left?
"""

solution = generator(prompt)
print(solution)
```
    

