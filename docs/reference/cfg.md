# Grammar-structured generation

You can pass any context-free grammar in the EBNF format and Outlines will generate an output that is valid to this grammar:

```python
from outlines import models, generate

arithmetic_grammar = """
    ?start: expression

    ?expression: term (("+" | "-") term)*

    ?term: factor (("*" | "/") factor)*

    ?factor: NUMBER
           | "-" factor
           | "(" expression ")"

    %import common.NUMBER
"""

model = models.transformers("WizardLM/WizardMath-7B-V1.1")
generator = generate.cfg(model, arithmetic_grammar)
sequence = generator(
  "Alice had 4 apples and Bob ate 2. "
  + "Write an expression for Alice's apples:"
)

print(sequence)
# (8-2)
```

!!! Note "Performance"

    The implementation of grammar-structured generation in Outlines is very naive. This does not reflect the performance of [.txt](https://dottxt.co)'s product, where we made grammar-structured generation as fast as regex-structured generation.


## Ready-to-use grammars

Outlines contains a (small) library of grammars that can be imported and use directly. We can rewrite the previous example as:

```python
from outlines import models, generate

arithmetic_grammar = outlines.grammars.arithmetic

model = models.transformers("WizardLM/WizardMath-7B-V1.1")
generator = generate.cfg(model, arithmetic_grammar)
sequence = generator(
  "Alice had 4 apples and Bob ate 2. "
  + "Write an expression for Alice's apples:"
)

print(sequence)
# (8-2)
```

The following grammars are currently available:

- Arithmetic grammar via `outlines.grammars.arithmetic`
- JSON grammar via `outlines.grammars.json`

If you would like more grammars to be added to the repository, please open an [issue](https://github.com/outlines-dev/outlines/issues) or a [pull request](https://github.com/outlines-dev/outlines/pulls).
