# Architecture Overview

This guide explains how Outlines works under the hood. Understanding the architecture helps contributors navigate the codebase and extend the library.

## Core Insight

Instead of generating text and hoping it matches a format, Outlines makes it impossible for the model to generate invalid outputs by masking invalid tokens during generation.

## Layer Stack

The library is organized in layers, from high-level user API to low-level token processing:

```
User API (outlines.from_transformers, outlines.from_openai, etc.)
    ↓
Generator Classes (SteerableGenerator, BlackBoxGenerator)
    ↓
Type System (types/dsl.py: Pydantic → JsonSchema → Regex)
    ↓
FSM Compilation (outlines-core: regex → FSM via interegular)
    ↓
Guide System (processors/guide.py: FSM state management)
    ↓
Logits Processing (processors/structured.py: token masking)
    ↓
Model Providers (transformers, OpenAI, etc.)
```

## Key Design Decisions

1. **FSM-based constraints**: For local models, constraints compile to finite state machines that track valid next tokens.

2. **Provider abstraction**: The same constraint system works across local models (transformers) and APIs (OpenAI).

3. **Lazy compilation**: FSMs are compiled on first use and cached persistently.

4. **Token-level control**: Constraints apply at the token level, not character level.

5. **Type-driven API**: Python types are the primary interface for specifying constraints.

## Core Components

### Models (`outlines/models/`)

Base classes and implementations for different model providers:

- **`SteerableModel`**: For models where we control logits (transformers, llama.cpp)
- **`BlackBoxModel`**: For API models with structured output support (OpenAI, Anthropic)

Each provider has an adapter class handling input and output format conversion.

Key files:

| File | Description |
|------|-------------|
| `base.py` | Abstract base classes defining the model interface |
| `transformers.py` | Integration with HuggingFace transformers |
| `openai.py` | OpenAI API integration |
| `ollama.py` | Ollama integration |
| `llamacpp.py` | llama.cpp integration |

### Generation (`outlines/generator.py`)

Handles the generation process with two main generator classes:

- **`BlackBoxGenerator`**: For API models with structured outputs support
- **`SteerableGenerator`**: For models where we control the logits

### FSM System (`outlines/fsm/` and `outlines/processors/`)

Core constraint enforcement:

- `processors/guide.py`: Base `Guide` class and `RegexGuide` implementation
- `fsm/parsing.py`: Lark-based CFG parsing with `PartialLark` parser

Key concepts:

- **Guide**: Manages FSM state during generation
- **State transitions**: Precomputed mapping of (state, token) → next_state
- **Token masking**: For each state, compute which tokens are valid

### Type System (`outlines/types/`)

Type conversion pipeline:

- `dsl.py`: Term DSL defining constraint language (Sequence, Choice, etc.)
- Python types → Term DSL → Regex → FSM

### Logits Processors (`outlines/processors/`)

Apply constraints during generation:

- `structured.py`: Main `StructuredLogitsProcessor`
- Processors mask invalid tokens by setting their logits to `-inf`

## How It Works

### FSM Compilation Pipeline

1. **Pattern definition**: User provides Pydantic model, regex, or grammar
2. **Schema to regex**: Convert complex types to regex patterns
   - JSON schemas become regex matching valid JSON
   - Pydantic models extract JSON schema then convert
3. **Regex to FSM**: Use interegular library to build FSM
4. **FSM to token map**: For each FSM state, compute valid tokens
5. **Guide creation**: Wrap FSM with state tracking

### Token Masking Process

```python
# Simplified logits processing
def process_logits(logits, current_state, guide):
    valid_tokens = guide.get_valid_tokens(current_state)
    mask = torch.full_like(logits, -float('inf'))
    mask[valid_tokens] = 0
    return logits + mask
```

## File Organization

```
outlines/
├── __init__.py              # Public API exports
├── generator.py             # Main Generator classes
├── models/                  # Model integrations
│   ├── base.py             # Abstract base classes
│   ├── transformers.py     # HuggingFace support
│   └── [provider].py       # Other providers
├── fsm/                     # FSM engine
│   └── parsing.py          # Grammar parsing
├── types/                   # Type system
│   ├── __init__.py         # Common regex types
│   └── dsl.py              # Term DSL and JSON schema conversion
├── processors/              # Logits processing and guides
│   ├── guide.py            # Guide implementations
│   ├── structured.py       # Main processor
│   └── tensor_adapters/    # Framework-specific tensor handling
└── caching.py               # Caching system
```

## Extension Points

### Adding a Model Provider

1. Create model class inheriting from `SteerableModel` or `BlackBoxModel`
2. Implement required methods: `generate()`, `generate_stream()`
3. Add constructor function in `outlines/__init__.py`
4. Handle provider-specific formats with a `TypeAdapter`

### Adding a Constraint Type

1. Define new Term subclass in `types/dsl.py`
2. Implement `to_regex()` conversion
3. Register type handler in `python_types_to_terms()`
4. Add tests for FSM compilation

### Custom Logits Processor

1. Inherit from `OutlinesLogitsProcessor`
2. Implement `process_logits()` method
3. Handle batch processing and state management
