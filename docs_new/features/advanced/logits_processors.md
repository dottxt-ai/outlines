---
title: Logits Processors
---

# Logits Processors

Logits processors are objects used to constrain the text generation of a model by applying biases to the logits associated with the tokens at each pass of the model, modifying the probability of each token occurring next in a sequence.
Processors can be used to:
1. Generate structured output, such as JSON
2. Disable certain words or tokens
3. Implement novel token sampling methods

## Overview

Outlines uses logits processors with so called steerable models, meaning models that run locally such that we have extensive control over the generation process. When using such models in Outlines, the output type provided is turned into a logits processor that is then passed on the inference engine.

There are four models for which we rely on logits processors:
- LlamaCpp
- MLXLM
- Transformers
- VLLMOffline

Instead of providing an output type that will be turned into a logits processor, it is possible to directly provide a logits processor. To do so, you must create a `Generator` instance using the `processor` keyword argument. You cannot directly call the model with a logits processor.

For instance:

```python
from outlines import Generator

model = ...
logits_processor = ...

# Create and call a generator
generator = Generator(model, processor=logits_processor)
response = generator("Hello work")
```

## Creating Custom Logits Processors

You can create your own custom logits processor by subclassing the `OutlinesLogitsProcessor` class. This allows you to implement specific logic to modify logits as needed.
Your logits processor needs to implement the `process_logits` method to modify the logits.
`process_logits` accepts
- `input_ids`: an array of token IDs generated so far.
- `logits`: vector of logits calculated by the language model's forward pass.

In the example below, we are creating a custom logits processor to force the model to provide a response that uses binary representation (so only the tokens for 0 and 1 are allowed):

```python
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor, TensorType
from outlines import Generator, from_transformers
import transformers

ALLOWED_TOKENS = [15, 16] # token ids for 0 and 1

class BinaryLogitsProcessor(OutlinesLogitsProcessor):

    def process_logits(self, input_ids: TensorType, logits: TensorType) -> TensorType:
        # Create a mask of all tokens
        mask = self.tensor_adapter.boolean_ones_like(logits)
        # Set mask to False for the allowed tokens
        for token_id in ALLOWED_TOKENS:
            mask[:, token_id] = False
        # Set non-allowed tokens to -inf so they are not selected
        logits[mask] = float("-inf")
        return logits

tf_tokenizer = transformers.AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B")
tf_model = transformers.AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Pro-Llama-3-8B")
model = from_transformers(tf_model, tf_tokenizer)

logits_processor = BinaryLogitsProcessor(model.tensor_library_name)

prompt = "Write the number 47 in binary. For example, 1010 is the binary representation of 10. Answer just with the binary number composed of 0s and 1s."
formatted_prompt = tf_tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False
)

generator = Generator(model, processor=logits_processor)
response = generator(formatted_prompt)

print(response) # "101111"
```
