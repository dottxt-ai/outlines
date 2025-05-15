---
title: Logits Processors
---

# Logits Processors

Logits processors are objects used to constrain the text generation of a model by applying biases to the logits associated to the tokens at each pass of the model.

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

You can create your own custom logits processor by subclassing the `OutlinesLogitsProcessor` class. This allows you to implement specific logic to modify logits as needed. Your logits processor needs to omplement the `process_logits` method to modify the logits. This method receives `input_ids` and `logits` and should return the modified logits.

For instance:

```python
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor, TensorType

class CustomLogitsProcessor(OutlinesLogitsProcessor):

    def process_logits(self, input_ids: TensorType, logits: TensorType) -> TensorType:
        forbidden_tokens = [token_id_1, token_id_2]
        for token_id in forbidden_tokens:
            logits[:, token_id] = float("-inf")
        return logits
```
