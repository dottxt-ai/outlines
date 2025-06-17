---
title: Logits Processors
---

# Logits Processors

Logits processors are objects that control text generation by modifying the probability distribution of possible next tokens. They do this by adjusting the logits (raw model outputs) at each generation step, effectively biasing the model's token selection.
Processors can be used to:
1. Generate structured output (e.g., JSON that follows a specific schema)
2. Prevent the model from generating specific words or tokens
3. Implement custom token sampling strategies

## Overview

Outlines uses logits processors with steerable models — models that run locally and allow fine-grained control over the generation process. When using such models in Outlines, the output type provided is turned into a logits processor that is then passed to the inference engine.

There are three models that support logits processors:

- LlamaCpp
- MLXLM
- Transformers

Instead of providing an output type that will be turned into a logits processor, it is possible to directly provide a logits processor. To do so, you must create a `Generator` instance using the `processor` keyword argument. You cannot directly call the model with a logits processor.

For instance:

```python
import transformers
from outlines import Generator, from_transformers
from outlines.processors import RegexLogitsProcessor

# Create a model
model = from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B"),
    transformers.AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B")
)

# Create a regex logits processor that only returns hex unicode notations
logits_processor = RegexLogitsProcessor(r"U\+[0-9A-Fa-f]{4,6}", model.tokenizer, model.tensor_library_name)

# Create a generator with the logits processor and use it to generate text
generator = Generator(model, processor=logits_processor)
response = generator("What's the unicode for the hugging face emoji")

print(response) # U+1F917
```

## Creating Custom Logits Processors

You can create your own logits processor by subclassing the `OutlinesLogitsProcessor` class. This allows you to implement specific logic to modify logits as needed.
Your logits processor needs to implement the `process_logits` method to modify the logits.
`process_logits` accepts:
- `input_ids`: the ids of the tokens of the existing sequences in a 2D tensor.
- `logits`: the logits for the current generation step in a 2D tensor.

In the example below, we create a custom logits processor to force the model to provide a response using only binary representation (so only the tokens for 0 and 1 are allowed):

```python
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor, TensorType
from outlines import Generator, from_transformers
import transformers

ALLOWED_TOKENS = [15, 16]  # token IDs corresponding to '0' and '1' in the model's vocabulary

# Subclass OutlinesLogitsProcessor
class BinaryLogitsProcessor(OutlinesLogitsProcessor):

    def process_logits(self, input_ids: TensorType, logits: TensorType) -> TensorType:
        # Create a mask for all tokens
        mask = self.tensor_adapter.boolean_ones_like(logits)
        # Set mask to False for the allowed tokens
        for token_id in ALLOWED_TOKENS:
            mask[:, token_id] = False
        # Set non-allowed tokens to -inf so they are not selected
        logits[mask] = float("-inf")
        return logits

# Create a regular model
tf_tokenizer = transformers.AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B")
tf_model = transformers.AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B")
model = from_transformers(tf_model, tf_tokenizer)

# Instantiate your custom logits processor
logits_processor = BinaryLogitsProcessor(model.tensor_library_name)

prompt = "Write the number 47 in binary. For example, 1010 is the binary representation of 10. Answer just with the binary number composed of 0s and 1s."
formatted_prompt = tf_tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False
)

# Create a generator with the custom logits processor instance and use it to generate text
generator = Generator(model, processor=logits_processor)
response = generator(formatted_prompt)

print(response) # "101111"
```
