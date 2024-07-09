# Thread-of-Thought (ThoT) Prompting


Thread-of-Thought (ThoT) Prompting is an improved thought inducer for chain-of-thought reasoning. Instead of using a simple prompt like "Let's think step by step," ThoT uses a more detailed thought inducer: "Walk me through this context in manageable parts step by step, summarizing and analyzing as we go." This technique is particularly effective in question-answering and retrieval settings, especially when dealing with large, complex contexts. It encourages the language model to break down the problem into smaller, more manageable parts and provide a more thorough analysis.
    

## A worked example


To implement Thread-of-Thought (ThoT) Prompting:

1. Start with your initial question or task.

2. Prepend the ThoT thought inducer to your prompt:
   "Walk me through this context in manageable parts step by step, summarizing and analyzing as we go."

3. Add your specific question or task after the thought inducer.

4. Submit the complete prompt to the language model.

5. The model should respond with a detailed, step-by-step analysis of the context and question.

Example:

User: Walk me through this context in manageable parts step by step, summarizing and analyzing as we go. What were the major causes of World War I?

AI: Certainly, I'll walk you through the major causes of World War I step by step, summarizing and analyzing as we go.

Step 1: Nationalism and Imperialism
Summary: In the early 20th century, European nations were driven by strong nationalist sentiments and imperialist ambitions.
Analysis: This created tensions as countries competed for colonial territories and sought to assert their dominance on the global stage.

Step 2: Militarism
Summary: There was an arms race among European powers, with countries building up their military forces.
Analysis: This increasing militarization created an atmosphere of mutual suspicion and fear, making conflict more likely.

Step 3: Alliance System
Summary: Europe was divided into two main alliances: the Triple Alliance (Germany, Austria-Hungary, Italy) and the Triple Entente (Britain, France, Russia).
Analysis: These alliances were intended to maintain a balance of power, but they also meant that a conflict between two countries could quickly escalate into a larger war.

Step 4: Assassination of Archduke Franz Ferdinand
Summary: The immediate trigger for the war was the assassination of Archduke Franz Ferdinand, heir to the Austro-Hungarian throne, by a Serbian nationalist in 1914.
Analysis: This event set off a chain reaction due to the complex system of alliances.

Conclusion: The major causes of World War I were a combination of long-term factors (nationalism, imperialism, militarism, and the alliance system) and the immediate trigger of the assassination. These factors together created a volatile situation that erupted into a global conflict.
    
## Code Example


```python
from typing import List
from pydantic import BaseModel, Field
import outlines

class AnalysisStep(BaseModel):
    step_number: int
    summary: str
    analysis: str

class ThoTAnalysis(BaseModel):
    steps: List[AnalysisStep]
    conclusion: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, ThoTAnalysis)

prompt = """Walk me through this context in manageable parts step by step, summarizing and analyzing as we go. What were the major causes of World War I?"""

analysis = generator(prompt)
print(analysis)

# Access individual steps
for step in analysis.steps:
    print(f"Step {step.step_number}:")
    print(f"Summary: {step.summary}")
    print(f"Analysis: {step.analysis}")
    print()

# Print conclusion
print(f"Conclusion: {analysis.conclusion}")
```
    

