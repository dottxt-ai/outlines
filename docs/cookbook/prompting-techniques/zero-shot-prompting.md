# Zero-Shot Prompting


Zero-Shot Prompting is a technique where you provide instructions to a language model without giving it any examples (exemplars) of the task. This approach relies on the model's pre-existing knowledge to interpret and respond to the prompt. The key is to clearly articulate the task, desired output format, and any specific constraints or requirements. Zero-Shot Prompting is particularly useful when you don't have access to relevant examples or when you want to test the model's base capabilities without additional context.
    

## A worked example


Let's say we want to use Zero-Shot Prompting to classify a movie review sentiment. Here's how to implement this technique step by step:

1. Formulate a clear instruction:
   Start with a concise explanation of the task. For example:
   "Classify the following movie review as positive or negative."

2. Specify the output format (optional but helpful):
   Add instructions on how you want the answer formatted. For example:
   "Respond with either 'Positive' or 'Negative' only."

3. Provide the input:
   Include the text to be classified. For example:
   "Movie review: The special effects were amazing, but the plot was confusing and the acting was terrible."

4. Combine all elements into a single prompt:
   "Classify the following movie review as positive or negative. Respond with either 'Positive' or 'Negative' only.

   Movie review: The special effects were amazing, but the plot was confusing and the acting was terrible."

5. Submit the prompt to the language model:
   Send this prompt to the AI model without any additional context or examples.

6. Receive and interpret the response:
   The model should respond with either "Negative" or "Positive" based on its interpretation of the review.

By following these steps, you've implemented Zero-Shot Prompting for sentiment analysis without providing any examples to the model.
    
## Code Example





```python
import outlines

model = outlines.models.transformers("google/gemma-2b")

prompt = """Classify the following movie review as positive or negative. Respond with either 'Positive' or 'Negative' only.

Movie review: The special effects were amazing, but the plot was confusing and the acting was terrible."""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
sentiment = generator(prompt)
print(f"The sentiment of the movie review is: {sentiment}")
```

    `config.hidden_act` is ignored, you should use `config.hidden_activation` instead.
    Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use
    `config.hidden_activation` if you want to override this behaviour.
    See https://github.com/huggingface/transformers/pull/29402 for more details.



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    The sentiment of the movie review is: Positive

