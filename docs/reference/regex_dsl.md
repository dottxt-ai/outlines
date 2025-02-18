# DSL to express constraints

This library provides a Domain-Specific Language (DSL) to construct regular expressions in a more intuitive and modular way. It allows you to create complex regexes using simple building blocks that represent literal strings, patterns, and various quantifiers. Additionally, these custom regex types can be used directly as types in [Pydantic](https://pydantic-docs.helpmanual.io/) schemas to enforce pattern constraints during text generation.

---

## Why Use This DSL?

1. **Modularity & Readability**: Instead of writing cryptic regular expression strings, you compose a regex as a tree of objects.
2. **Enhanced Debugging**: Each expression can be visualized as an ASCII tree, making it easier to understand and debug complex regexes.
3. **Pydantic Integration**: Use your DSL-defined regex as types in Pydantic models. The DSL seamlessly converts to JSON Schema with proper pattern constraints.
4. **Extensibility**: Easily add or modify quantifiers and other regex components by extending the provided classes.

---

## Building Blocks


Every regex component in this DSL is a **Term**. Here are two primary types:

- **`String`**: Represents a literal string. It escapes the characters that have a special meaning in regular expressions.
- **`Regex`**: Represents an existing regex pattern string.

```python
from outlines.types import String, Regex

# A literal string "hello"
literal = String("hello")   # Internally represents "hello"

# A regex pattern to match one or more digits
digit = Regex(r"[0-9]+")     # Internally represents the pattern [0-9]+

# Converting to standard regex strings:
from outlines.types.dsl import to_regex

print(to_regex(literal))  # Output: hello
print(to_regex(digit))    # Output: [0-9]+
```

---

## Early Introduction to Quantifiers & Operators

The DSL supports common regex quantifiers as methods on every `Term`. These methods allow you to specify how many times a pattern should be matched. They include:

- **`times(count)`**: Matches the term exactly `count` times.
- **`optional()`**: Matches the term zero or one time.
- **`one_or_more()`**: Matches the term one or more times (Kleene Plus).
- **`zero_or_more()`**: Matches the term zero or more times (Kleene Star).
- **`repeat(min_count, max_count)`**: Matches the term between `min_count` and `max_count` times (or open-ended if one value is omitted).

Let’s see these quantifiers side by side with examples.

### Quantifiers in Action

#### `times(count)`

This method restricts the term to appear exactly `count` times.

```python
# Example: exactly 5 digits
five_digits = Regex(r"\d").times(5)
print(to_regex(five_digits))  # Output: (\d){5}
```

You can also use the `times` function:

```python
from outlines.types import times

# Example: exactly 5 digits
five_digits = times(Regex(r"\d"), 5)
print(to_regex(five_digits))  # Output: (\d){5}
```

#### `optional()`

The `optional()` method makes a term optional, meaning it may occur zero or one time.

```python
# Example: an optional "s" at the end of a word
maybe_s = String("s").optional()
print(to_regex(maybe_s))  # Output: (s)?
```

You can also use the `optional` function, the string will automatically be converted to a `String` object:

```python
from outlines.types import optional

# Example: an optional "s" at the end of a word
maybe_s = optional("s")
print(to_regex(maybe_s))  # Output: (s)?
```

#### `one_or_more()`

This method indicates that the term must appear at least once.

```python
# Example: one or more alphabetic characters
letters = Regex(r"[A-Za-z]").one_or_more()
print(to_regex(letters))  # Output: ([A-Za-z])+
```

You can also use the `one_or_more` function:

```python
from outlines.types import one_or_more

# Example: one or more alphabetic characters
letters = one_or_more(Regex(r"[A-Za-z]"))
print(to_regex(letters))  # Output: ([A-Za-z])+

```

#### `zero_or_more()`

This method means that the term can occur zero or more times.

```python
# Example: zero or more spaces
spaces = String(" ").zero_or_more()
print(to_regex(spaces))  # Output: ( )*
```

You can also use the `zero_or_more` function, the string will automatically be converted to a `String` instance:

```python
from outlines.types import zero_or_more

# Example: zero or more spaces
spaces = zero_or_more(" ")
print(to_regex(spaces))  # Output: ( )*
```

#### `repeat(min_count, max_count)`

The `repeat` method provides flexibility to set a lower and/or upper bound on the number of occurrences.

```python
# Example: Between 2 and 4 word characters
word_chars = Regex(r"\w").repeat(2, 4)
print(to_regex(word_chars))  # Output: (\w){2,4}

# Example: At least 3 digits (min specified, max left open)
at_least_three = Regex(r"\d").repeat(3, None)
print(to_regex(at_least_three))  # Output: (\d){3,}

# Example: Up to 2 punctuation marks (max specified, min omitted)
up_to_two = Regex(r"[,.]").repeat(None, 2)
print(to_regex(up_to_two))  # Output: ([,.]){,2}
```

You can also use the `repeat` function:

```python
from outlines import repeat

# Example: Between 2 and 4 word characters
word_chars = repeat(Regex(r"\w"), 2, 4)
print(to_regex(word_chars))  # Output: (\w){2,4}

# Example: At least 3 digits (min specified, max left open)
at_least_three = repeat(Regex(r"\d"), 3, None)
print(to_regex(at_least_three))  # Output: (\d){3,}

# Example: Up to 2 punctuation marks (max specified, min omitted)
up_to_two = repeat(Regex(r"[,.]"), None, 2)
print(to_regex(up_to_two))  # Output: ([,.]){,2}
```

---

## Combining Terms

The DSL allows you to combine basic terms into more complex patterns using concatenation and alternation.

### Concatenation (`+`)

The `+` operator (and its reflected variant) concatenates terms, meaning that the terms are matched in sequence.

```python
# Example: Match "hello world"
pattern = String("hello") + " " + Regex(r"\w+")
print(to_regex(pattern))  # Output: hello\ (\w+)
```

### Alternation (`|`)

The `|` operator creates alternatives, allowing a match for one of several patterns.

```python
# Example: Match either "cat" or "dog"
animal = String("cat") | "dog"
print(to_regex(animal))  # Output: (cat|dog)
```

*Note:* When using operators with plain strings (such as `"dog"`), the DSL automatically wraps them in a `String` object and escapes the characters that have a special meaning in regular expressions.

---

## Custom types

The DSL comes "batteries included" with types that represent common text constructs:

- `integer` represents an integer number as recognized by `int`
- `boolean` represents a boolean, "True" or "False" as recognized by `bool`
- `number` represents a floating-point number recognize by Python's `float`
- `date` represents a date as understood by `datetime.date`
- `time` represents a time as undestood by `datetime.time`
- `datetime` represents a time as understoof by `datetime.datetime`
- `digit` represents a single digit
- `char` represents a single character
- `newline` represents a new line character
- `whitespace` represents a white space
- `sentence` represents a sentence
- `paragraph` reprensents a pagraph (one or more sentences separated by one or more ilne breaks)


For instance you can describe the answers in the GSM8K dataset using the following pattern:

```python
from outlines.types import sentence, digit

answer = "A: " + sentence.repeat(2,4) + " So the answer is: " + digit.repeat(1,4)
```

---

## Practical Examples

### Example 1: Matching a Custom ID Format

Suppose you want to create a regex that matches an ID format like "ID-12345", where:
- The literal "ID-" must be at the start.
- Followed by exactly 5 digits.

```python
id_pattern = "ID-" + Regex(r"\d").times(5)
print(to_regex(id_pattern))  # Output: ID-(\d){5}
```

### Example 2: Email Validation with Pydantic

You can define a regex for email validation and use it as a type in a Pydantic model.

```python
from pydantic import BaseModel, ValidationError

# Define an email regex term (this is a simplified version)
email_regex = Regex(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

class User(BaseModel):
    name: str
    email: email_regex  # Use our DSL regex as a field type

# Valid input
user = User(name="Alice", email="alice@example.com")
print(user)

# Invalid input (raises a ValidationError)
try:
    User(name="Bob", email="not-an-email")
except ValidationError as e:
    print(e)
```

When used in a Pydantic model, the email field is automatically validated against the regex pattern and its JSON Schema includes the `pattern` constraint.

### Example 3: Building a Complex Pattern

Consider a pattern to match a simple date format: `YYYY-MM-DD`.

```python
year = Regex(r"\d").times(4)         # Four digits for the year
month = Regex(r"\d").times(2)         # Two digits for the month
day = Regex(r"\d").times(2)           # Two digits for the day

# Combine with literal hyphens
date_pattern = year + "-" + month + "-" + day
print(to_regex(date_pattern))
# Output: (\d){4}\-(\d){2}\-(\d){2}
```

---

## Visualizing Your Pattern

One of the unique features of this DSL is that each term can print its underlying structure as an ASCII tree. This visualization can be particularly helpful when dealing with complex expressions.

```python
# A composite pattern using concatenation and quantifiers
pattern = "a" + String("b").one_or_more() + "c"
print(pattern)
```

*Expected Output:*

```
└── Sequence
    ├── String('a')
    ├── KleenePlus(+)
    │   └── String('b')
    └── String('c')
```

This tree representation makes it easy to see the hierarchy and order of operations in your regular expression.

---

## Final Words

This DSL is designed to simplify the creation and management of regular expressions—whether you're validating inputs in a web API, constraining the output of an LLM, or just experimenting with regex patterns. With intuitive methods for common quantifiers and operators, clear visual feedback, and built-in integration with Pydantic, you can build robust and maintainable regex-based validations with ease.

Feel free to explore the library further and adapt the examples to your use cases. Happy regexing!
