---
title: Few-Shot Prompting
---

# Few-Shot Prompting


Few-Shot Prompting is a technique where you provide the AI model with a small number of examples (typically 2-5) demonstrating the task you want it to perform, followed by the actual task you want completed. This approach helps the model understand the context and format of the desired output without requiring fine-tuning. 

The key to effective Few-Shot Prompting is selecting relevant examples that closely match the structure and complexity of your target task. These examples act as a form of implicit instruction, guiding the model's behavior and improving its performance on the specific task at hand.

To implement Few-Shot Prompting:
1. Identify the task you want the model to perform.
2. Create 2-5 example input-output pairs that demonstrate the task.
3. Format these examples consistently, typically using an "Input: [text] Output: [response]" structure.
4. Place these examples at the beginning of your prompt, followed by your actual task.
5. Ensure there's a clear delineation between the examples and your target task.

This technique is particularly useful for tasks where the model might struggle with zero-shot performance, or when you need to specify a particular output format or style.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).
## A worked example


Let's implement Few-Shot Prompting for a simple sentiment analysis task:

1. Identify the task: Classify movie reviews as positive or negative.

2. Create example input-output pairs:
Example 1:
Input: This movie was absolutely fantastic! I loved every minute of it.
Output: Positive

Example 2:
Input: I've never been so bored in my life. Complete waste of time.
Output: Negative

Example 3:
Input: The film had its moments, but overall it was pretty average.
Output: Neutral

3. Format the prompt with examples and the target task:

Here are some examples of movie review sentiment analysis:

Input: This movie was absolutely fantastic! I loved every minute of it.
Output: Positive

Input: I've never been so bored in my life. Complete waste of time.
Output: Negative

Input: The film had its moments, but overall it was pretty average.
Output: Neutral

Now, please analyze the sentiment of this movie review:

Input: The special effects were incredible, but the plot made no sense at all.
Output:

4. Send this prompt to the AI model. The model will likely respond with a sentiment classification based on the examples provided, such as:

Output: Mixed

This example demonstrates how Few-Shot Prompting can guide the model to perform sentiment analysis with nuanced categories (positive, negative, neutral, mixed) based on the provided examples.
    
## Code Example






```python
from pydantic import BaseModel, Field
from typing import List
import outlines

class MovieReview(BaseModel):
    sentiment: str = Field(..., description="Overall sentiment of the review (Positive, Negative, or Mixed)")
    rating: int = Field(..., ge=1, le=10, description="Rating out of 10")
    key_points: List[str] = Field(..., max_items=3, description="Up to 3 key points from the review")

model = outlines.models.transformers("google/gemma-2b")

prompt = """Analyze movie reviews and provide structured output. Here are some examples:

Review: "This film was a masterpiece! The acting was superb, and the plot kept me on the edge of my seat. However, the pacing was a bit slow at times."
Output: {"sentiment": "Positive", "rating": 9, "key_points": ["Masterpiece", "Superb acting", "Slow pacing"]}

Review: "I expected so much more. The special effects were decent, but the story was confusing and the characters were poorly developed."
Output: {"sentiment": "Negative", "rating": 4, "key_points": ["Decent special effects", "Confusing story", "Poor character development"]}

Review: "An average movie with some good moments. The lead actor gave a strong performance, but the plot was predictable."
Output: {"sentiment": "Mixed", "rating": 6, "key_points": ["Average overall", "Strong lead performance", "Predictable plot"]}

Now, analyze this review:
Review: "The visuals were breathtaking and the sound design was immersive. Unfortunately, the dialogue felt forced and unnatural, which took me out of the experience."
Output:"""

generator = outlines.generate.json(model, MovieReview)
result = generator(prompt)
print(result)
```

    `config.hidden_act` is ignored, you should use `config.hidden_activation` instead.
    Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use
    `config.hidden_activation` if you want to override this behaviour.
    See https://github.com/huggingface/transformers/pull/29402 for more details.



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|██████████| 90/90 [00:04<00:00, 19.72it/s]


    sentiment='Negative' rating=2 key_points=['Anyone', 'Statute', 'Immersive sound design']



```python

```
