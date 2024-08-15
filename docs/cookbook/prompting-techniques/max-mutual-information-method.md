---
title: Max Mutual Information Method
---

# Max Mutual Information Method


The Max Mutual Information Method is a prompting technique that aims to find the most effective prompt template by creating multiple variations and selecting the one that maximizes the mutual information between the prompt and the language model's outputs. This approach involves generating diverse prompt templates with different styles and exemplars, using each template to generate outputs from the language model, calculating the mutual information between the prompts and their corresponding outputs, and finally selecting the template that yields the highest mutual information score. By doing so, this method seeks to identify the prompt that elicits the most informative and relevant responses from the language model.

Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).
## Step by Step Example


Let's walk through a simple example of using the Max Mutual Information Method for a sentiment analysis task:

1. Create diverse prompt templates:
   Template A: "Analyze the sentiment of the following text: [TEXT]"
   Template B: "Is the following statement positive, negative, or neutral? [TEXT]"
   Template C: "Rate the emotional tone of this sentence from 1 (very negative) to 5 (very positive): [TEXT]"

2. Generate outputs for each template using a set of sample texts:
   Sample text: "I love this new restaurant!"

   Template A output: "The sentiment of the text is positive."
   Template B output: "The statement is positive."
   Template C output: "4 - The emotional tone is quite positive."

3. Calculate mutual information:
   (This step would typically involve complex calculations, but for simplicity, we'll use a simplified scoring method)
   Template A score: 0.7
   Template B score: 0.8
   Template C score: 0.6

4. Select the template with the highest mutual information:
   Based on our simplified scores, Template B has the highest mutual information score.

5. Use the selected template for future prompts:
   For subsequent sentiment analysis tasks, we would use Template B: "Is the following statement positive, negative, or neutral? [TEXT]"

This process helps identify the most effective prompt template for the specific task and language model being used.

## Code Example






```python
import outlines

model = outlines.models.transformers("google/gemma-2b")

# Sample text for sentiment analysis
sample_text = "I love this new restaurant!"

# Define prompt templates
templates = [
    f"Analyze the sentiment of the following text: {sample_text}",
    f"Is the following statement positive, negative, or neutral? {sample_text}",
    f"Rate the emotional tone of this sentence as positive, negative, or neutral: {sample_text}"
]

# Generate outputs for each template
generator = outlines.generate.choice(model, ["Positive", "Negative", "Neutral"])
outputs = [generator(template) for template in templates]

# Simulate mutual information calculation (simplified scoring)
scores = [0.7, 0.8, 0.6]

# Select the best template
best_template_index = scores.index(max(scores))
best_template = templates[best_template_index]

print(f"Best template: {best_template}")
print(f"Output using best template: {outputs[best_template_index]}")
```

    `config.hidden_act` is ignored, you should use `config.hidden_activation` instead.
    Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use
    `config.hidden_activation` if you want to override this behaviour.
    See https://github.com/huggingface/transformers/pull/29402 for more details.



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Best template: Is the following statement positive, negative, or neutral? I love this new restaurant!
    Output using best template: Negative
