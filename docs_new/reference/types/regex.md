---
title: Regular Expressions
---

# Regular Expressions

::: outlines.types.regex

Regular expressions provide a way to constrain the model's output to match specific text patterns.

## Basic Usage

```python
from outlines import Generator, from_transformers
from outlines.types import Regex

# Create a regex pattern for a North American phone number
phone_regex = Regex(r"\(\d{3}\) \d{3}-\d{4}")

# Initialize a model
model = from_transformers(...)

# Create a generator with the regex output type
generator = Generator(model, phone_regex)

# Generate a phone number
prompt = "Generate a random phone number in the format (XXX) XXX-XXXX:"
result = generator(prompt)

print(result)  # Example output: "(555) 123-4567"
```

## Complex Patterns

```python
from outlines import Generator, from_transformers
from outlines.types import Regex

# Create a regex pattern for an email address
email_regex = Regex(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Initialize a model
model = from_transformers(...)

# Create a generator with the regex output type
generator = Generator(model, email_regex)

# Generate an email address
prompt = "Generate a professional email address for John Doe:"
result = generator(prompt)

print(result)  # Example output: "john.doe@example.com"
```

## Creating a Regex Type

The `Regex` class accepts a string containing a regular expression pattern:

```python
from outlines.types import Regex

# Simple pattern for a 5-digit US ZIP code
zip_code = Regex(r"\d{5}")

# Pattern for a date in YYYY-MM-DD format
date_format = Regex(r"\d{4}-\d{2}-\d{2}")

# Pattern for a URL
url_pattern = Regex(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9./_-]*)?")
```

For more detailed examples, see the [Regular Expression guide](/user_guide/structured_generation/regex.md).
