# Compressing prompts with SuperCompress

Long prompts make structured generation slower, because every input token has to go through the model before the constrained output starts. If most of the context is irrelevant to the question, you can trim it before calling Outlines.

[SuperCompress](https://github.com/arjunkshah/supercompress) is a small, CPU-only library that keeps the lines of a long context that matter for a given query, under a token budget. It is an optional, third-party preprocessing step: Outlines does not depend on it, and you only need it if your inputs are long enough for the trimming to pay off.

See the [SuperCompress guide](https://supercompress.dev/token-compression) for how the compression works and how its output compares to simple truncation.

## When it helps

- The context is long (documents, logs, transcripts) and only a few passages answer the question.
- You run structured generation (JSON schemas, regex, grammars) on that context many times, so the prefill cost adds up.

It does not help much when the whole input is needed, or when the prompt is already short.

## Install

SuperCompress is not a dependency of Outlines. Install it next to Outlines:

```bash
pip install supercompress
```

## Example

This example compresses a long document for a query, then generates a JSON object that matches a Pydantic schema.

```python
import outlines
import transformers
from pydantic import BaseModel
from supercompress import compress_context


class Order(BaseModel):
    pizza: str
    number: int


model_name = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda"),
    transformers.AutoTokenizer.from_pretrained(model_name),
)

long_context = open("orders_export.txt").read()
question = "Which pizza was ordered, and how many?"

result = compress_context(long_context, question, budget_ratio=0.35)

prompt = (
    "Extract the order from the notes below as JSON.\n\n"
    f"NOTES:\n{result.compressed_text}\n\n"
    "RESULT: "
)

generator = outlines.Generator(model, Order)
order = Order.model_validate_json(generator(prompt))
print(order)
```

`budget_ratio` is the fraction of the original tokens to keep. Lower values compress more and may drop relevant text, so check the output on a few of your own inputs before you rely on a particular ratio.

The generator returns the raw JSON text. Parse it with the Pydantic model, as shown above.

## Notes

- `compress_context` runs locally and needs PyTorch, which the package installs as a dependency. SuperCompress also offers a hosted API, which requires an API key; the example above does not use it.
- Compression is lossy. Keep the original context if you need an exact quotation or a complete audit trail.
