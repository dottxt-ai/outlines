# transformers


!!! Installation

    You need to install the `transformer` and `datasets` libraries to be able to use these models in Outlines.


Outlines provides an integration with the `torch` implementation of causal models in the [transformers][transformers] library. You can initialize the model by passing its name:

```python
from outlines import models

model = models.transformers("mistralai/Mistral-7B-v0.1", device="cuda")
```

If you need more fine-grained control you can also initialize the model and tokenizer separately:


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import models

llm = AutoModelForCausalLM.from_pretrained("gpt2", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = models.Transformers(llm, tokenizer)
```

[transformers]: https://github.com/huggingface/transformers
