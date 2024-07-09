# Zero-Shot Prompting


Zero-Shot Prompting is a technique where a language model is given a task or question without any examples or training specific to that task. The model relies solely on its pre-trained knowledge to generate a response. This technique tests the model's ability to understand and perform tasks based on natural language instructions alone, without the benefit of task-specific examples.
    

## A worked example


To implement Zero-Shot Prompting:

1. Identify the task you want the model to perform.
2. Craft a clear, concise instruction that describes the task without providing any examples.
3. Include any necessary context or constraints for the task.
4. Submit the prompt to the language model.
5. Evaluate the model's response.

For example, if you want the model to classify a movie review sentiment:

1. Task: Sentiment analysis of a movie review
2. Craft instruction: "Determine whether the following movie review is positive or negative. Respond with only 'Positive' or 'Negative'."
3. Include context: "Movie review: The special effects were amazing, but the plot was confusing and the acting was terrible."
4. Submit the full prompt to the model:
   "Determine whether the following movie review is positive or negative. Respond with only 'Positive' or 'Negative'.
   Movie review: The special effects were amazing, but the plot was confusing and the acting was terrible."
5. Evaluate the model's response (which should be either "Positive" or "Negative").

This example demonstrates Zero-Shot Prompting because it provides no examples of sentiment analysis, relying entirely on the model's pre-existing understanding of movie reviews and sentiment to perform the task.
    
## Code Example


```python
import outlines

# Set up the model
model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

# Create the prompt
prompt = """Determine whether the following movie review is positive or negative.

Movie review: The special effects were amazing, but the plot was confusing and the acting was terrible."""

# Generate the response
generator = outlines.generate.choice(model, ["Positive", "Negative"])
sentiment = generator(prompt)

print(f"The sentiment of the review is: {sentiment}")
```
    

