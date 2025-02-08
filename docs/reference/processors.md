# Logit Processors

Logit processors modify token probabilities during text generation to enforce constraints or analyze the generation process. While processors can be used directly, most users will interact with them through the high-level generation APIs (see [Generating JSON](generation/json.md), [Regex Generation](generation/regex.md), and [CFG Generation](generation/cfg.md)).

## Available Processors

Outlines provides several specialized processors for different use cases:

- `JSONLogitsProcessor`: Ensures generation follows a JSON schema
- `RegexLogitsProcessor`: Constrains generation to match a regex pattern
- `CFGLogitsProcessor`: Enforces a context-free grammar
- `LogitTrackingProcessor`: Analyzes token probabilities during generation
- `GuideLogitsProcessor`: Base class for guided generation

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

There are two ways to add tracking to a generator:

```python
from outlines.generate import json
from outlines.processors import add_tracking, LogitTrackingProcessor
from pydantic import BaseModel

# Create a generator
class WeatherResponse(BaseModel):
    temperature: float
    conditions: str

generator = json(model, WeatherResponse)

# Method 1: Using the convenience function (recommended)
generator = add_tracking(generator)  # Enable tracking
# Or with a sliding window:
generator = add_tracking(generator, max_positions=5)  # Only track last 5 positions

# Method 2: Using the processor directly
generator.logits_processor = LogitTrackingProcessor(generator.logits_processor)
# Or with a sliding window:
generator.logits_processor = LogitTrackingProcessor(generator.logits_processor, max_positions=5)
```

### Analyzing generation results

Once tracking is enabled, you can analyze the generation process by calling these methods on the generator:

- `get_statistics(pos)`: Get basic statistics about the logits at a position
- `get_top_tokens(pos, k=5)`: View the most likely tokens at a position
- `get_sequence(end_pos, use_processed=True)`: Reconstruct the generated string from tracked logits

```python
# Generate response
response = generator("What's the weather like?")

# Check token probabilities at each step
for pos in range(3):
    stats = generator.logits_processor.get_statistics(pos)
    tokens = generator.logits_processor.get_top_tokens(pos, k=5)
    print(f"\nPosition {pos}:")
    print(f"Valid tokens: {stats['processed']['valid_tokens']}")
    print("Top tokens:", [t['token'] for t in tokens['processed']])
```

### Memory management

The tracking processor stores logits in memory for analysis. For long sequences, you have two options:

1. Use a sliding window to limit memory usage:
```python
# Only track the 5 most recent positions
generator = add_tracking(generator, max_positions=5)
```

2. Clear tracking data when no longer needed:
```python
generator.logits_processor.clear_tracking()
```

### Important notes

- Tracking logits is a slow operation, so do not use it in production environments
- The processor stores logits in memory, so consider using `max_positions` for long sequences
- Processed logits will contain `-inf` values when structured outputs are used
- Token decoding requires the wrapped processor to have a tokenizer attribute
- Memory usage grows linearly with sequence length when `max_positions` is not set

## Advanced Usage

### Custom Processors

To create a custom processor, inherit from `OutlinesLogitsProcessor`:

```python
from outlines.processors import OutlinesLogitsProcessor

class MyProcessor(OutlinesLogitsProcessor):
    def process_logits(self, input_ids, logits):
        # Modify logits here
        return logits
```

### GuideLogitsProcessor

The `GuideLogitsProcessor` is a base class for processors that use a guide (like regex or CFG) to constrain generation:

```python
from outlines.processors import GuideLogitsProcessor
from outlines.fsm.guide import RegexGuide

# Create a guide
guide = RegexGuide.from_regex(r"[0-9]{4}", tokenizer)

# Create processor with guide
processor = GuideLogitsProcessor(tokenizer=tokenizer, guide=guide)
```

### Troubleshooting

Common issues when using processors:

1. Memory usage with tracking:
   - Use `max_positions` to limit memory usage
   - Clear tracking data regularly with `clear_tracking()`

2. Performance issues:
   - Avoid using tracking in production
   - Consider using simpler constraints when possible

3. Invalid outputs:
   - Check that the processor's constraints are correct
   - Use tracking to analyze where generation goes wrong
