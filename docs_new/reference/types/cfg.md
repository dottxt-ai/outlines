---
title: Context-Free Grammars
---

# Context-Free Grammars

::: outlines.types.cfg

Context-free grammars (CFGs) provide a powerful way to define complex structured outputs. They're particularly useful for generating text with a specific syntax, such as code snippets or custom formats.

## Basic Usage

```python
from outlines import Generator, from_transformers
from outlines.types import CFG

# Define a grammar for simple arithmetic expressions
grammar = """
start: expr
expr: term "+" expr | term
term: factor "*" term | factor
factor: NUMBER | "(" expr ")"
NUMBER: /[0-9]+/
"""

# Create a CFG output type
cfg_output = CFG(grammar)

# Initialize a model
model = from_transformers(...)

# Create a generator with the CFG output type
generator = Generator(model, cfg_output)

# Generate an arithmetic expression
prompt = "Generate a simple arithmetic expression:"
result = generator(prompt)

print(result)  # Example output: "3 * (4 + 5)"
```

## Grammar Definition

Context-free grammars in Outlines use the [Lark grammar format](https://lark-parser.readthedocs.io/en/latest/grammar/). A grammar consists of rules that define the structure of the text you want to generate.

```python
# Grammar for a simple SQL query
sql_grammar = """
start: select_stmt
select_stmt: "SELECT" columns "FROM" table ("WHERE" condition)?
columns: column ("," column)*
column: IDENTIFIER
table: IDENTIFIER
condition: IDENTIFIER comparison_op value
comparison_op: "=" | ">" | "<" | ">=" | "<=" | "!="
value: NUMBER | STRING
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
NUMBER: /[0-9]+/
STRING: /"[^"]*"/
"""
```

## Advanced Usage

Context-free grammars can be used for complex generation tasks like code generation:

```python
from outlines import Generator, from_transformers
from outlines.types import CFG

# Grammar for simple Python functions
python_grammar = """
start: function
function: "def" IDENTIFIER "(" params? ")" ":" NEWLINE INDENT body DEDENT
params: param ("," param)*
param: IDENTIFIER
body: statement+
statement: assignment NEWLINE | if_stmt | return_stmt NEWLINE
assignment: IDENTIFIER "=" expr
if_stmt: "if" expr ":" NEWLINE INDENT body DEDENT
return_stmt: "return" expr
expr: IDENTIFIER | NUMBER | STRING | expr "+" expr | expr "-" expr
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
NUMBER: /[0-9]+/
STRING: /"[^"]*"/
INDENT: /\t/
DEDENT: /\t/
NEWLINE: /\n/
"""

# Create a CFG output type
cfg_output = CFG(python_grammar)

# Initialize a model
model = from_transformers(...)

# Create a generator
generator = Generator(model, cfg_output)

# Generate a Python function
prompt = "Generate a Python function that calculates the factorial of a number:"
result = generator(prompt)

print(result)
```

For more detailed examples, see the [Context-Free Grammar guide](/user_guide/structured_generation/cfg.md).

## Notes

- Context-Free Grammar support in Outlines is experimental but powerful
- Complex grammars may take longer to process
- For simple structures, consider using regex or JSON schema instead
