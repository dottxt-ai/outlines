# JSON mode

Outlines can guarantee that the LLM will generate valid JSON, using [Grammar-structured generation](cfg.md):

```python
from outlines import models, generate

json_grammar = outlines.grammars.json

model = models.transformers("mistralai/Mistral-7b-v0.1")
generator = generate.cfg(model, json_grammar)
sequence = generator("Generate valid JSON")
```

!!! Note "JSON that follows a schema"

    If you want to guarantee that the generated JSON follows a given schema, consult [this section](json.md) instead.
