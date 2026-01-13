# Architecture Overview

This guide helps you understand how Outlines is organized so you can navigate the codebase, debug issues, and extend the library when needed.

## How Structured Generation Works

When you ask an LLM to output JSON or follow a specific format, traditional approaches generate text freely and hope it matches. Outlines takes a different approach: it constrains the model at generation time by masking invalid tokens, making it impossible for the model to produce invalid output.

## Core Abstractions

Outlines has three main abstractions that work together: **Model**, **Generator**, and **Type System**. Understanding these will help you navigate most of the codebase.

### Model and ModelTypeAdapter

The `Model` class (`outlines/models/base.py`) represents any LLM that Outlines can work with. There are two categories:

**Why two model types?**

- **`SteerableModel`**: Models where we have direct access to logits (token probabilities). This includes local models like HuggingFace Transformers or llama.cpp. We can modify the logits to mask invalid tokens before sampling.

- **`BlackBoxModel`**: API-based models (OpenAI, Anthropic) where we cannot access logits. These models have their own structured output features that we delegate to.

**The Model interface:**

Every model implements these methods:

| Method | Purpose |
|--------|---------|
| `__call__(prompt, output_type)` | Generate a single response |
| `batch(prompts, output_type)` | Generate responses for multiple prompts |
| `stream(prompt, output_type)` | Stream a response token by token |
| `generate(...)` | Internal method that subclasses implement |

**ModelTypeAdapter - Bridging formats:**

Each model has a `type_adapter` attribute that handles format conversion between Outlines and the specific model provider:

```python
class ModelTypeAdapter(ABC):
    def format_input(self, model_input) -> Any:
        """Convert user input to model-specific format.

        For API models: creates the `messages` argument
        For local models: may convert str to list, apply chat templates, etc.
        """
        ...

    def format_output_type(self, output_type) -> Any:
        """Convert output type to model-specific format.

        For API models: creates `response_format` argument
        For local models: creates logits processor
        """
        ...
```

This abstraction allows the same user code to work across different model providers without changes.

### Generator - Unifying the Generation Interface

The `Generator` (`outlines/generator.py`) is a factory function that creates the appropriate generator class based on the model type. It exists to solve a key problem: **the same user code should work regardless of whether you're using a local model or an API**.

**Why Generator exists:**

Without Generator, users would need to write different code for different model types:

```python
# Without Generator - user needs to know model internals
if isinstance(model, SteerableModel):
    processor = build_logits_processor(output_type)
    result = model.generate(prompt, processor)
else:
    result = model.generate(prompt, output_type)
```

With Generator, the complexity is hidden:

```python
# With Generator - same code works for any model
generator = Generator(model, output_type)
result = generator(prompt)
```

**Generator classes:**

| Class | Used For | How It Works |
|-------|----------|--------------|
| `SteerableGenerator` | Local models | Builds and caches a logits processor from the output type, passes it to model |
| `BlackBoxGenerator` | API models | Passes output type directly to model's structured output feature |

**SteerableGenerator internals:**

When you create a `SteerableGenerator` with an output type, it:

1. Converts the Python type to a `Term` (regex DSL representation)
2. Builds a logits processor from that term
3. Caches the processor for reuse
4. On each call, resets processor state and passes it to the model

```python
class SteerableGenerator:
    def __init__(self, model, output_type, backend_name=None):
        self.model = model
        # Convert type -> Term -> logits processor
        term = python_types_to_terms(output_type)
        if isinstance(term, CFG):
            self.logits_processor = get_cfg_logits_processor(...)
        elif isinstance(term, JsonSchema):
            self.logits_processor = get_json_schema_logits_processor(...)
        else:
            regex_string = to_regex(term)
            self.logits_processor = get_regex_logits_processor(...)

    def __call__(self, prompt, **kwargs):
        self.logits_processor.reset()  # Reset state for new generation
        return self.model.generate(prompt, self.logits_processor, **kwargs)
```

### Type System - From Python Types to Constraints

The type system (`outlines/types/dsl.py`) converts Python types into constraints that can be enforced during generation.

**The conversion pipeline:**

```
Python Type → Term (DSL) → Regex/JsonSchema/CFG → FSM → Logits Processor
```

**Term classes:**

`Term` is the base class for Outlines' constraint DSL. Key subclasses:

| Term | Purpose | Example |
|------|---------|---------|
| `Regex` | Match a regex pattern | `Regex("[0-9]+")` |
| `JsonSchema` | Match valid JSON for a schema | `JsonSchema(MyModel)` |
| `CFG` | Match a context-free grammar | `CFG(json_grammar)` |
| `Sequence` | Concatenate terms | `String("[") + item + String("]")` |
| `Alternatives` | Match any of several terms | `term1 \| term2` |

**python_types_to_terms:**

This function is the entry point for type conversion:

```python
def python_types_to_terms(ptype) -> Term:
    if isinstance(ptype, Term):
        return ptype
    elif is_int(ptype):
        return types.integer  # Predefined regex for integers
    elif is_pydantic_model(ptype):
        return JsonSchema(ptype)
    elif is_enum(ptype):
        return Alternatives([...])  # One term per enum value
    # ... more type handlers
```

## Data Flow

Here's how a typical structured generation request flows through the system:

```
1. User calls: model("What is 2+2?", int)

2. Model.__call__ creates Generator:
   Generator(model, int)

3. Generator (SteerableGenerator) builds processor:
   - python_types_to_terms(int) → Regex("-?[0-9]+")
   - get_regex_logits_processor(...) → LogitsProcessor

4. Generator calls model.generate(prompt, processor)

5. During generation, for each token:
   - Model computes logits for all tokens
   - LogitsProcessor masks invalid tokens (set to -inf)
   - Model samples from remaining valid tokens

6. Result returned to user
```

## File Organization

```
outlines/
├── __init__.py              # Public API: from_transformers, from_openai, etc.
├── generator.py             # Generator factory and classes
├── models/
│   ├── base.py             # Model, AsyncModel, ModelTypeAdapter base classes
│   ├── transformers.py     # HuggingFace Transformers integration
│   ├── openai.py           # OpenAI API integration
│   └── ...                 # Other providers
├── types/
│   ├── __init__.py         # Predefined types: integer, number, date, etc.
│   └── dsl.py              # Term classes and python_types_to_terms
├── processors/
│   ├── structured.py       # LogitsProcessor implementations
│   └── ...
├── backends/               # Backend-specific logits processor builders
└── fsm/                    # FSM and grammar parsing utilities
```

## Extension Points

### Adding a New Model Provider

1. Create a new file in `outlines/models/`
2. Implement `Model` (or `AsyncModel`) subclass with `generate`, `generate_batch`, `generate_stream`
3. Implement `ModelTypeAdapter` subclass with `format_input`, `format_output_type`
4. Add factory function (e.g., `from_mymodel`) in `outlines/__init__.py`

### Adding a New Output Type

1. If it can be expressed as regex: add handler in `python_types_to_terms()`
2. If it needs JSON schema: ensure it can be converted to `JsonSchema` term
3. If it needs grammar: create `CFG` term with grammar definition

### Custom Logits Processor

1. Inherit from the base processor class in `outlines/processors/`
2. Implement the `process_logits()` method
3. Handle state reset in `reset()` method
