# Few-Shot Prompting


Few-Shot Prompting is a technique where you provide the AI model with a small number of examples (usually 2-5) of the task you want it to perform, followed by a new query. This allows the model to learn from the examples and apply that knowledge to the new query, often improving performance compared to zero-shot prompting. The examples typically follow a consistent format, showing input-output pairs that demonstrate the desired behavior.
    

## A worked example


To implement Few-Shot Prompting:

1. Choose 2-5 representative examples of the task you want the model to perform. These should be diverse enough to cover different aspects of the task.

2. Format each example as an input-output pair, typically using a consistent structure like "Input: [example input] Output: [example output]".

3. Combine these examples into a prompt, separating them with line breaks.

4. Add your actual query at the end of the prompt, using the same format as the examples but leaving the output blank.

5. Submit this entire prompt to the AI model.

Here's a specific example for a sentiment analysis task:

Input: The movie was absolutely terrible. I hated every minute of it.
Output: Negative

Input: I had a great time at the restaurant. The food was delicious and the service was excellent.
Output: Positive

Input: The book was okay. It had some interesting parts, but overall it was pretty average.
Output: Neutral

Input: I just got back from my vacation and it was amazing!
Output:

In this example, we've provided three input-output pairs demonstrating sentiment analysis, followed by a new input for the model to classify. The model should use the patterns in the examples to determine that the correct output for the final input is "Positive".
    
## Code Example


```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

prompt = """You are a sentiment analysis assistant. Classify the sentiment of the following reviews as Positive, Negative, or Neutral.

Review: The movie was absolutely terrible. I hated every minute of it.
Sentiment: Negative

Review: I had a great time at the restaurant. The food was delicious and the service was excellent.
Sentiment: Positive

Review: The book was okay. It had some interesting parts, but overall it was pretty average.
Sentiment: Neutral

Review: I just got back from my vacation and it was amazing!
Sentiment: """

generator = outlines.generate.choice(model, ["Positive", "Negative", "Neutral"])
sentiment = generator(prompt)
print(f"The sentiment of the last review is: {sentiment}")
```
    

