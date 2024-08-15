---
title: Demonstration Ensembling (DENSE)
---

# Demonstration Ensembling (DENSE)


DENSE is a prompting technique that creates multiple few-shot prompts, each containing a distinct subset of exemplars from the training set. It then generates responses for each prompt and aggregates these outputs to produce a final response. This approach helps reduce variance in LLM outputs and can improve accuracy by leveraging diverse examples.


## Step by Step Example


Let's implement DENSE for a simple sentiment analysis task on movie reviews:

1. Select training examples:
   - "A masterpiece!" (Positive)
   - "Absolutely terrible." (Negative)
   - "Enjoyable but forgettable." (Neutral)
   - "Waste of time." (Negative)
   - "Highly recommended!" (Positive)

2. Create multiple prompts with different subsets:

Prompt 1:
Classify the sentiment of movie reviews as Positive, Negative, or Neutral.

Examples:
"A masterpiece!" - Positive
"Waste of time." - Negative

Review: "I couldn't stop watching."

Prompt 2:
Classify the sentiment of movie reviews as Positive, Negative, or Neutral.

Examples:
"Absolutely terrible." - Negative
"Highly recommended!" - Positive

Review: "I couldn't stop watching."

Prompt 3:
Classify the sentiment of movie reviews as Positive, Negative, or Neutral.

Examples:
"Enjoyable but forgettable." - Neutral
"Highly recommended!" - Positive

Review: "I couldn't stop watching."

3. Generate responses for each prompt:
   Prompt 1 output: Positive
   Prompt 2 output: Positive
   Prompt 3 output: Positive

4. Aggregate results:
   Final output: Positive (3/3 prompts agree)

This example demonstrates how DENSE uses multiple prompts with different example subsets to generate a more robust final classification.

Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## Code Example






```python
import outlines
from pydantic import BaseModel, Field
from typing import List, Literal

# Define our Pydantic models
class Example(BaseModel):
    review: str
    sentiment: Literal["Positive", "Negative", "Neutral"]

class PromptResult(BaseModel):
    prompt_id: int
    result: Literal["Positive", "Negative", "Neutral"]

class DENSEResult(BaseModel):
    prompt_results: List[PromptResult]
    final_result: Literal["Positive", "Negative", "Neutral"]

# Initialize the model
model = outlines.models.transformers("google/gemma-2b")

# Define our examples
examples = [
    Example(review="A masterpiece!", sentiment="Positive"),
    Example(review="Absolutely terrible.", sentiment="Negative"),
    Example(review="Enjoyable but forgettable.", sentiment="Neutral"),
    Example(review="Waste of time.", sentiment="Negative"),
    Example(review="Highly recommended!", sentiment="Positive")
]

# Function to create a prompt with a subset of examples
def create_prompt(example_subset, review_to_classify):
    prompt = "Classify the sentiment of movie reviews as Positive, Negative, or Neutral.\n\nExamples:\n"
    for example in example_subset:
        prompt += f'"{example.review}" - {example.sentiment}\n'
    prompt += f'\nReview: "{review_to_classify}"'
    return prompt

# Create multiple prompts with different subsets of examples
prompts = [
    create_prompt([examples[0], examples[3]], "I couldn't stop watching."),
    create_prompt([examples[1], examples[4]], "I couldn't stop watching."),
    create_prompt([examples[2], examples[4]], "I couldn't stop watching.")
]

# Generate results for each prompt
generator = outlines.generate.choice(model, ["Positive", "Negative", "Neutral"])
results = [generator(prompt) for prompt in prompts]

# Create the DENSE result
dense_result = DENSEResult(
    prompt_results=[PromptResult(prompt_id=i+1, result=result) for i, result in enumerate(results)],
    final_result=max(set(results), key=results.count)
)

# Generate the final structured output
json_generator = outlines.generate.json(model, DENSEResult)
final_output = json_generator(str(dense_result))

print(final_output)
```

    `config.hidden_act` is ignored, you should use `config.hidden_activation` instead.
    Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use
    `config.hidden_activation` if you want to override this behaviour.
    See https://github.com/huggingface/transformers/pull/29402 for more details.



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]



Compiling FSM index for all state transitions:   0%|                                                                                                                                  | 0/119 [00:00<?, ?it/s]


Compiling FSM index for all state transitions:   1%|█                                                                                                                         | 1/119 [00:00<00:20,  5.82it/s]


Compiling FSM index for all state transitions:   4%|█████▏                                                                                                                    | 5/119 [00:00<00:05, 20.21it/s]


Compiling FSM index for all state transitions:   8%|█████████▏                                                                                                                | 9/119 [00:00<00:04, 27.04it/s]


Compiling FSM index for all state transitions:  11%|█████████████▏                                                                                                           | 13/119 [00:00<00:03, 30.80it/s]


Compiling FSM index for all state transitions:  14%|█████████████████▎                                                                                                       | 17/119 [00:00<00:03, 33.53it/s]


Compiling FSM index for all state transitions:  18%|█████████████████████▎                                                                                                   | 21/119 [00:00<00:02, 33.73it/s]


Compiling FSM index for all state transitions:  21%|█████████████████████████▍                                                                                               | 25/119 [00:00<00:02, 33.25it/s]


Compiling FSM index for all state transitions:  24%|█████████████████████████████▍                                                                                           | 29/119 [00:00<00:02, 34.06it/s]


Compiling FSM index for all state transitions:  28%|█████████████████████████████████▌                                                                                       | 33/119 [00:01<00:02, 34.82it/s]


Compiling FSM index for all state transitions:  31%|█████████████████████████████████████▌                                                                                   | 37/119 [00:01<00:02, 35.70it/s]


Compiling FSM index for all state transitions:  34%|█████████████████████████████████████████▋                                                                               | 41/119 [00:01<00:02, 34.98it/s]


Compiling FSM index for all state transitions:  38%|█████████████████████████████████████████████▊                                                                           | 45/119 [00:01<00:02, 34.80it/s]


Compiling FSM index for all state transitions:  41%|█████████████████████████████████████████████████▊                                                                       | 49/119 [00:01<00:01, 35.38it/s]


Compiling FSM index for all state transitions:  45%|█████████████████████████████████████████████████████▉                                                                   | 53/119 [00:01<00:01, 35.64it/s]


Compiling FSM index for all state transitions:  48%|█████████████████████████████████████████████████████████▉                                                               | 57/119 [00:01<00:01, 36.37it/s]


Compiling FSM index for all state transitions:  51%|██████████████████████████████████████████████████████████████                                                           | 61/119 [00:01<00:01, 36.89it/s]


Compiling FSM index for all state transitions:  55%|██████████████████████████████████████████████████████████████████                                                       | 65/119 [00:01<00:01, 36.42it/s]


Compiling FSM index for all state transitions:  58%|██████████████████████████████████████████████████████████████████████▏                                                  | 69/119 [00:02<00:01, 36.61it/s]


Compiling FSM index for all state transitions:  61%|██████████████████████████████████████████████████████████████████████████▏                                              | 73/119 [00:02<00:01, 36.96it/s]


Compiling FSM index for all state transitions:  65%|██████████████████████████████████████████████████████████████████████████████▎                                          | 77/119 [00:02<00:01, 37.43it/s]


Compiling FSM index for all state transitions:  68%|██████████████████████████████████████████████████████████████████████████████████▎                                      | 81/119 [00:02<00:01, 36.89it/s]


Compiling FSM index for all state transitions:  71%|██████████████████████████████████████████████████████████████████████████████████████▍                                  | 85/119 [00:02<00:01, 33.59it/s]


Compiling FSM index for all state transitions:  75%|██████████████████████████████████████████████████████████████████████████████████████████▍                              | 89/119 [00:02<00:00, 34.77it/s]


Compiling FSM index for all state transitions:  78%|██████████████████████████████████████████████████████████████████████████████████████████████▌                          | 93/119 [00:02<00:00, 35.50it/s]


Compiling FSM index for all state transitions:  82%|██████████████████████████████████████████████████████████████████████████████████████████████████▋                      | 97/119 [00:02<00:00, 35.85it/s]


Compiling FSM index for all state transitions:  85%|█████████████████████████████████████████████████████████████████████████████████████████████████████▊                  | 101/119 [00:02<00:00, 35.29it/s]


Compiling FSM index for all state transitions:  88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▉              | 105/119 [00:03<00:00, 35.86it/s]


Compiling FSM index for all state transitions:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▉          | 109/119 [00:03<00:00, 36.10it/s]


Compiling FSM index for all state transitions:  95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉      | 113/119 [00:03<00:00, 36.21it/s]


Compiling FSM index for all state transitions:  98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉  | 117/119 [00:03<00:00, 36.30it/s]


Compiling FSM index for all state transitions: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 119/119 [00:03<00:00, 34.51it/s]




    prompt_results=[PromptResult(prompt_id=1, result='Positive'), PromptResult(prompt_id=2, result='Positive'), PromptResult(prompt_id=3, result='Negative'), PromptResult(prompt_id=4, result='Negative')] final_result='Neutral'
