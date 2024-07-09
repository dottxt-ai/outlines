# Step-Back Prompting


Step-Back Prompting is a modification of Chain-of-Thought (CoT) prompting that aims to improve reasoning on complex questions. It works by first asking the language model a high-level, abstract question about relevant concepts or background knowledge before having it tackle the specific problem at hand. This "stepping back" to consider the broader context helps prime the model with relevant information and frameworks, potentially leading to better reasoning and more accurate answers on the subsequent specific question.
    

## A worked example


To implement Step-Back Prompting:

1. Start with your original question, e.g.:
"What is the orbital period of a satellite 400 km above Earth's surface?"

2. Formulate a high-level, abstract question related to the topic:
"What are the key factors and principles involved in determining a satellite's orbital period?"

3. Construct your prompt with both questions:

"I'm going to ask you a specific question, but first, please answer this general question:
What are the key factors and principles involved in determining a satellite's orbital period?

After you've answered that, please address this specific question:
What is the orbital period of a satellite 400 km above Earth's surface?"

4. Submit this two-part prompt to the language model.

5. The model will first provide general information about satellite orbits, then use that context to reason about and answer the specific question.

This technique allows the model to "step back" and consider relevant principles before tackling the specific problem, potentially leading to more accurate and well-reasoned responses.
    
## Code Example


```python
from pydantic import BaseModel, constr
import outlines

class CharacterBackstory(BaseModel):
    general_principles: constr(max_length=500)
    character_name: constr(max_length=30)
    backstory: constr(max_length=300)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, CharacterBackstory)

prompt = """I'm going to ask you about a specific character's backstory, but first, please answer this general question:
What are the key elements to consider when creating a compelling character backstory for a role-playing game?

After you've answered that, please address this specific request:
Create a brief backstory for a character named Lyra who is a skilled archer living in a medieval fantasy world."""

backstory = generator(prompt)
print(backstory)
```
    

