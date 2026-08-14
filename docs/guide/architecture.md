# Architecture Overview

This guide explains how Outlines is organized so you can navigate the codebase, debug issues, and extend the library.

## How Structured Generation Works

When you ask an LLM to output JSON or follow a specific format, traditional approaches generate text freely and hope it matches. Outlines takes a different approach: it constrains the model at generation time by masking invalid tokens, making it impossible for the model to produce invalid output.

## Core Abstractions

Outlines has three main abstractions: **Model**, **Generator**, and **Type System**.

### Model and ModelTypeAdapter

The `Model` class (`outlines/models/base.py`) is the abstract base class for all LLM integrations. There are two categories based on how structured generation is implemented:

**Steerable models** (`SteerableModel`): Models where Outlines directly applies a logits processor during generation. This includes:
- `LlamaCpp` - llama.cpp bindings
- `MLXLM` - Apple MLX models
- `Transformers` - HuggingFace Transformers

**Black-box models** (`BlackBoxModel`): Models where Outlines delegates structured generation to the provider's API rather than applying logits processors directly. This includes:
- `OpenAI`, `Anthropic`, `Gemini`, `Mistral` - Cloud API providers
- `VLLM`, `VLLMOffline`, `SGLang`, `TGI`, `Ollama` - Inference servers with built-in structured generation
- `Dottxt` - Dottxt API

Note: Some black-box models (like vLLM or Ollama) could technically expose logits, but they implement structured generation server-side, so Outlines delegates to their APIs instead of building processors locally.

**The Model interface:**

Every model subclass must implement these methods:

| Method | Purpose |
|--------|---------|
| `generate(model_input, output_type, **kwargs)` | Generate a single response (internal, receives logits processor or output type) |
| `generate_batch(model_input, output_type, **kwargs)` | Generate responses for multiple prompts |
| `generate_stream(model_input, output_type, **kwargs)` | Stream a response token by token |

The base `Model` class provides these convenience methods that create a `Generator` internally:

| Method | Purpose |
|--------|---------|
| `__call__(model_input, output_type, backend, **kwargs)` | Generate a single response |
| `batch(model_input, output_type, backend, **kwargs)` | Generate batch responses |
| `stream(model_input, output_type, backend, **kwargs)` | Stream a response |

**ModelTypeAdapter - Bridging formats:**

Each model has a `type_adapter` attribute that handles format conversion between Outlines and the specific model provider:

```python
class ModelTypeAdapter(ABC):
    @abstractmethod
    def format_input(self, model_input) -> Any:
        """Convert user input to model-specific format.

        For API models: creates the `messages` argument
        For local models: may apply chat templates, convert str to list, etc.
        """
        ...

    @abstractmethod
    def format_output_type(self, output_type) -> Any:
        """Convert output type to model-specific format.

        For black-box models: creates `response_format` argument
        For steerable models: formats the logits processor for the model
        """
        ...
```

### Generator - Unifying the Generation Interface

The `Generator` (`outlines/generator.py`) is a factory function that returns the appropriate generator class based on the model type.

**Why Generator exists:**

Without Generator, users would need different code for different model types:

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
| `SteerableGenerator` | Local models (`LlamaCpp`, `MLXLM`, `Transformers`) | Builds and caches a logits processor from the output type, resets and passes it to the model on each call |
| `BlackBoxGenerator` | Sync API models | Passes output type directly to model's generate method |
| `AsyncBlackBoxGenerator` | Async API models | Async version of BlackBoxGenerator |

**SteerableGenerator internals:**

When you create a `SteerableGenerator` with an output type, it:

1. Converts the Python type to a `Term` using `python_types_to_terms()`
2. Based on the Term type, builds the appropriate logits processor:
   - `CFG` → calls `get_cfg_logits_processor()`
   - `JsonSchema` → calls `get_json_schema_logits_processor()`
   - Other terms → converts to regex via `to_regex()`, then calls `get_regex_logits_processor()`
3. Caches the processor for reuse
4. On each call, resets processor state and passes it to the model

### Type System - From Python Types to Constraints

The type system (`outlines/types/dsl.py`) converts Python types into constraints that can be enforced during generation.

**The conversion pipeline:**

```
Python Type → Term (via python_types_to_terms)
                    ↓
            ┌───────┴───────┐
            ↓               ↓
    CFG or JsonSchema    Other Terms
            ↓               ↓
    Direct to backend   to_regex() → Regex string
            ↓               ↓
            └───────┬───────┘
                    ↓
            Logits Processor (via backend)
```

**Term classes:**

`Term` is the base class for Outlines' constraint DSL. Key subclasses:

| Term | Purpose | Example |
|------|---------|---------|
| `Regex` | Match a regex pattern | `Regex("[0-9]+")` |
| `JsonSchema` | Match valid JSON for a schema | `JsonSchema(MyPydanticModel)` |
| `CFG` | Match a context-free grammar | `CFG(grammar_string)` |
| `String` | Match a literal string | `String("hello")` |
| `Sequence` | Concatenate terms | `String("[") + item + String("]")` |
| `Alternatives` | Match any of several terms | `term1 \| term2` |
| `KleeneStar` | Zero or more repetitions | `zero_or_more(term)` |
| `KleenePlus` | One or more repetitions | `one_or_more(term)` |
| `Optional` | Zero or one occurrence | `optional(term)` |

**python_types_to_terms:**

This function converts Python types to Term instances:

```python
def python_types_to_terms(ptype) -> Term:
    # Already a Term - return as-is
    if isinstance(ptype, Term):
        return ptype

    # Basic types - return predefined regex patterns
    if is_int(ptype):
        return types.integer
    if is_float(ptype):
        return types.number
    if is_str(ptype):
        return types.string
    if is_bool(ptype):
        return types.boolean

    # Structured types - convert to JsonSchema
    if is_pydantic_model(ptype) or is_dataclass(ptype) or is_typed_dict(ptype):
        return JsonSchema(ptype)

    # Enum - create alternatives from members
    if is_enum(ptype):
        return Alternatives([...])

    # Union, Literal, List, Tuple, Dict - handle recursively
    ...
```

## Data Flow

Here's how a structured generation request flows through the system:

```
1. User calls: model("What is 2+2?", int)

2. Model.__call__ creates Generator:
   Generator(model, int)

3. Generator factory checks model type:
   - SteerableModel → SteerableGenerator
   - BlackBoxModel → BlackBoxGenerator

4. For SteerableGenerator:
   a. python_types_to_terms(int) → Regex("-?[0-9]+")
   b. to_regex(term) → regex string
   c. get_regex_logits_processor(backend, model, regex) → LogitsProcessor

5. Generator.__call__(prompt):
   a. processor.reset()  # Reset state for new generation
   b. model.generate(prompt, processor)

6. During generation (steerable models only):
   - Model computes logits for all tokens
   - LogitsProcessor masks invalid tokens (set to -inf)
   - Model samples from remaining valid tokens

7. Result returned to user
```

## File Organization

```
outlines/
├── __init__.py              # Public API exports
├── generator.py             # Generator factory and classes
├── models/
│   ├── base.py              # Model, AsyncModel, ModelTypeAdapter base classes
│   ├── transformers.py      # HuggingFace Transformers
│   ├── llamacpp.py          # llama.cpp bindings
│   ├── mlxlm.py             # Apple MLX models
│   ├── openai.py            # OpenAI API
│   ├── anthropic.py         # Anthropic API
│   ├── vllm.py              # vLLM server
│   ├── vllm_offline.py      # vLLM offline mode
│   └── ...                  # Other providers
├── types/
│   ├── __init__.py          # Predefined types: integer, number, date, etc.
│   ├── dsl.py               # Term classes, python_types_to_terms, to_regex
│   └── utils.py             # Type checking utilities
├── backends/
│   ├── __init__.py          # get_*_logits_processor functions
│   ├── base.py              # LogitsProcessorType protocol
│   ├── outlines_core.py     # Default backend using outlines-core
│   ├── llguidance.py        # Microsoft llguidance backend
│   └── xgrammar.py          # xgrammar backend
├── processors/
│   ├── base_logits_processor.py  # Base processor implementation
│   └── tensor_adapters/     # Tensor library adapters
├── grammars/                # Predefined grammar files
└── templates.py             # Prompt template utilities
```

## Backends

Backends are responsible for converting constraints (regex, JSON schema, CFG) into logits processors that can be applied during generation. They only apply to steerable models.

**Available backends:**

| Backend | Default For | Description |
|---------|-------------|-------------|
| `outlines_core` | Regex, JSON Schema | The default backend, built on the `outlines-core` Rust library. Compiles constraints into finite state machines. |
| `llguidance` | CFG | Microsoft's llguidance library. Supports context-free grammars and is the only backend that handles CFG constraints. |
| `xgrammar` | - | Alternative backend using the xgrammar library. |

**How backends are selected:**

1. If the user specifies a backend via the `backend` parameter, that backend is used
2. Otherwise, the default backend for the constraint type is used:
   - Regex → `outlines_core`
   - JSON Schema → `outlines_core`
   - CFG → `llguidance`

**Backend interface:**

All backends inherit from `BaseBackend` and implement three methods:

```python
class BaseBackend(ABC):
    @abstractmethod
    def get_json_schema_logits_processor(self, json_schema: str) -> LogitsProcessorType:
        ...

    @abstractmethod
    def get_regex_logits_processor(self, regex: str) -> LogitsProcessorType:
        ...

    @abstractmethod
    def get_cfg_logits_processor(self, grammar: str) -> LogitsProcessorType:
        ...
```

**Specifying a backend:**

```python
from outlines import from_transformers, Generator

model = from_transformers("microsoft/Phi-3-mini-4k-instruct")

# Use xgrammar instead of the default outlines_core
generator = Generator(model, int, backend="xgrammar")
```

## Extension Points

### Adding a New Model Provider

1. Create a new file in `outlines/models/` (e.g., `mymodel.py`)
2. Implement a `ModelTypeAdapter` subclass with `format_input()` and `format_output_type()`
3. Implement a `Model` subclass with `generate()`, `generate_batch()`, and `generate_stream()`
4. Add a factory function (e.g., `from_mymodel()`)
5. Export from `outlines/models/__init__.py`
6. Add to `SteerableModel` or `BlackBoxModel` type alias as appropriate
