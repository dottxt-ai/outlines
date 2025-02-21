# Logit processors

Logit processors modify token probabilities during text generation to enforce constraints or analyze the generation process. While processors can be used directly, most users will interact with them through the high-level generation APIs (see [Generating JSON](generation/json.md), [Regex Generation](generation/regex.md), and [CFG Generation](generation/cfg.md)).

Users can track the token probabilities and logits at each step of the generation process using the `LogitTrackingProcessor`. This is useful for debugging and understanding the generation process.

## Available Processors

Outlines provides several specialized processors for different use cases:

- `JSONLogitsProcessor`: Ensures generation follows a JSON schema
- `RegexLogitsProcessor`: Constrains generation to match a regex pattern
- `CFGLogitsProcessor`: Enforces a context-free grammar
- `LogitTrackingProcessor`: Tracks token probabilities and logits

### RegexLogitsProcessor

The `RegexLogitsProcessor` constrains generation to match a regular expression pattern:

```python
from outlines.processors import RegexLogitsProcessor

# Create a processor that only allows 4-digit numbers
processor = RegexLogitsProcessor(r"[0-9]{4}", tokenizer)

# Use with a generator
generator = outlines.generate.regex(model, r"[0-9]{4}")
generator.logits_processor = processor
```

See [Regex Generation](generation/regex.md) for more details and examples.

### JSONLogitsProcessor

The `JSONLogitsProcessor` ensures generation follows a JSON schema defined using Pydantic:

```python
from pydantic import BaseModel
from outlines.processors import JSONLogitsProcessor

class Response(BaseModel):
    name: str
    age: int
    city: str

# Create processor from schema
processor = JSONLogitsProcessor(Response, tokenizer)

# Use with a generator
generator = outlines.generate.json(model, Response)
generator.logits_processor = processor
```

See [Generating JSON](generation/json.md) for more details and examples.

### CFGLogitsProcessor

The `CFGLogitsProcessor` constrains generation to follow a context-free grammar:

```python
from outlines.processors import CFGLogitsProcessor

# Define a simple grammar
grammar = """
start: NUMBER "+" NUMBER "=" NUMBER
NUMBER: /[0-9]+/
"""

# Create processor from grammar
processor = CFGLogitsProcessor(grammar, tokenizer)

# Use with a generator
generator = outlines.generate.cfg(model, grammar)
generator.logits_processor = processor
```

See [CFG Generation](generation/cfg.md) for more details and examples.

## Tracking logit scores and token probabilities

The `LogitTrackingProcessor` wraps any processor to track logit scores and token probabilities before and after processing. This is useful for:

- Debugging logit processors by analyzing how they modify token probabilities
- Visualizing the effects of logit biasing on token distributions
- Understanding how constraints affect the generation process
- Validating that processors are working as intended

### Adding tracking to a generator

The simplest way to add tracking is using the convenience function `add_tracking`:

```python
from outlines import generate, models
from outlines.processors import add_tracking
from pydantic import BaseModel

# Define your schema
class Person(BaseModel):
    name: str
    age: int

# Create generator with tracking
model = models.transformers("HuggingFaceTB/SmolLM2-135M-Instruct")
generator = generate.json(model, Person)
generator = add_tracking(generator)  # Enable tracking

# Apply templating if needed
prompt = model.tokenizer.tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant, responding in JSON."},
     {"role": "user", "content": "Make me a person with a name and age. Return the JSON only."}],
    tokenize=False,
    add_bos=True,
    add_generation_prompt=True,
)

# Generate the response
response = generator(prompt)
```

**NOTE**: You __must__ use `generator.logits_processor.clear()` between generations, otherwise the processor will use the logits from the previous generation. You may also construct a new generator and call `add_tracking` again to start tracking from scratch.

### Analyzing generation results

Once tracking is enabled, you can analyze the generation process in several ways:

1. Get the top tokens at each position:

```python
# Get top 5 tokens at each position
top_k = generator.logits_processor.get_top_tokens(k=5)

# Analyze each position
for position_dict in top_k:
    print(f"\nPosition {position_dict['position']}:")
    print(f"Text so far: {position_dict['text_so_far']}")

    for token in position_dict['tokens']:
        print(f"\nToken: {token['token']}")
        print(f"Unstructured probability: {token['unstructured_prob']:.3f}")
        print(f"Structured probability: {token['structured_prob']:.3f}")
        print(f"Unstructured logit: {token['unstructured_logit']:.3f}")
        print(f"Structured logit: {token['structured_logit']:.3f}")
        print(f"Was chosen: {token['is_chosen']}")
```

2. Convert to a pandas DataFrame for analysis:

```python
import pandas as pd

# Get all tokens with probability > 1%
df = generator.logits_processor.to_dataframe(show="probs", min_value=0.01)
print(df)
#    position token   natural  constrained  chosen
# 0         0   You  0.021324          0.0   False
# 1         0   The  0.021959          0.0   False
# 2         0  Sure  0.025492          0.0   False
# 3         0  JSON  0.031045          0.0   False
# 4         0    To  0.031047          0.0   False
```

3. Get the generated sequence up to a position:

```python
# Get text generated up to position 5
text = generator.logits_processor.sequence(5)
```

### Memory management

The tracking processor stores logits in memory for analysis, and offloads logits to main memory if you use a GPU. For long sequences, you have several options:

1. Clear tracking data when no longer needed:
```python
generator.logits_processor.clear()
```

2. Filter data when analyzing:
```python
# Only analyze specific positions
results = generator.logits_processor.get_top_tokens(positions=[0, 1, 2])

# Only look at high probability tokens
df = generator.logits_processor.to_dataframe(show="probs", min_value=0.01)
```

### Important notes about logit tracking

- Tracking logits is a slow operation, so do not use it in production environments
- The processor will accumulate logits if you call `generator(prompt)` multiple times, meaning that the tokens stored can be aggregated across generations. You can use `generator.logits_processor.clear()` to reset the processor, or construct a new generator and call `add_tracking` again to start tracking from scratch.
- Processed logits will contain `-inf` values when structured outputs are used
- Token decoding requires the wrapped processor to have a tokenizer attribute
- Memory usage grows linearly with sequence length
- The tracking processor only supports single-batch processing
- Tracking logits can incur significant overhead -- do not use it in production environments

## Using the tracking processor directly

The tracking processor can be used directly with transformers pipelines:

```python
import outlines.models as models
import transformers
from outlines.processors import RegexLogitsProcessor
from outlines.processors.tracking import LogitTrackingProcessor

model_uri = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = models.transformers(model_uri)
    
outlines_tokenizer = models.TransformerTokenizer(
    transformers.AutoTokenizer.from_pretrained(model_uri)
)
phone_number_logits_processor = LogitTrackingProcessor(RegexLogitsProcessor(
    "\\+?[1-9][0-9]{7,14}",  # phone number pattern
    outlines_tokenizer,
))

generator = transformers.pipeline('text-generation', model=model_uri)

# Perform inference
output = generator(
    "Jenny gave me her number it's ",
    logits_processor=transformers.LogitsProcessorList([phone_number_logits_processor])
)

# Retrieve the logits
phone_number_logits_processor.get_logits()
```
