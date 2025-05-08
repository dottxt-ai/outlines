---
title: Output Types
---

# Output Types

Outlines provides several ways to define structured outputs for language models.

## Available Output Types

### JSON Schema

JSON Schema allows you to define the structure of JSON data that you want to generate. Outlines supports multiple ways to define JSON schemas:

- JSON Schema objects
- Pydantic models
- TypedDict
- Dataclasses

[Learn more about JSON Schema →](json_schema.md)

### Regular Expressions

Regular expressions provide a way to constrain the model's output to match specific text patterns. This is useful for generating formatted data like dates, phone numbers, or other structured text.

[Learn more about Regular Expressions →](regex.md)

### Context-Free Grammars

Context-free grammars (CFGs) provide the most flexible way to define complex structured outputs. They're particularly useful for generating text with a specific syntax, such as code or custom formats.

[Learn more about Context-Free Grammars →](cfg.md)

## How Output Types Work

Internally, Outlines converts output type definitions into finite state machines (FSMs) that constrain the model's generation process. This happens automatically when you pass an output type to the `Generator` class.
