import json as json
import re
from dataclasses import dataclass
from typing import Any, List, Union

from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema as cs
from outlines_core.fsm.json_schema import build_regex_from_schema


class Term:
    """Represents types defined with a regular expression.

    `Regex` instances can be used as a type in a Pydantic model definittion.
    They will be translated to JSON Schema as a "string" field with the
    "pattern" keyword set to the regular expression this class represents. The
    class also handles validation.

    Examples
    --------

    >>> from outlines.types import Regex
    >>> from pydantic import BaseModel
    >>>
    >>> age_type = Regex("[0-9]+")
    >>>
    >>> class User(BaseModel):
    >>>     name: str
    >>>     age: age_type

    """

    def __add__(self: "Term", other: Union[str, "Term"]) -> "Sequence":
        if isinstance(other, str):
            other = String(other)

        return Sequence([self, other])

    def __radd__(self: "Term", other: Union[str, "Term"]) -> "Sequence":
        if isinstance(other, str):
            other = String(other)

        return Sequence([other, self])

    def __get_validator__(self, _core_schema):
        def validate(input_value):
            return self.validate(input_value)

        return validate

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> cs.CoreSchema:
        return cs.no_info_plain_validator_function(lambda value: self.validate(value))

    def __get_pydantic_json_schema__(
        self, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string", "pattern": to_regex(self)}

    def validate(self, value: str) -> str:
        pattern = to_regex(self)
        compiled = re.compile(pattern)
        if not compiled.fullmatch(str(value)):
            raise ValueError(
                f"Input should be in the language of the regular expression {pattern}"
            )
        return value

    def matches(self, value: str) -> bool:
        """Check that a given value is in the language defined by the Term.

        We make the assumption that the language defined by the term can
        be defined with a regular expression.

        """
        pattern = to_regex(self)
        compiled = re.compile(pattern)
        if compiled.fullmatch(str(value)):
            return True
        return False

    def display_ascii_tree(self, indent="", is_last=True) -> str:
        """Display the regex tree in ASCII format."""
        branch = "└── " if is_last else "├── "
        result = indent + branch + self._display_node() + "\n"

        # Calculate the new indent for children
        new_indent = indent + ("    " if is_last else "│   ")

        # Let each subclass handle its children
        result += self._display_children(new_indent)
        return result

    def _display_node(self):
        raise NotImplementedError

    def _display_children(self, indent: str) -> str:
        """Display the children of this node. Override in subclasses with children."""
        return ""

    def __str__(self):
        return self.display_ascii_tree()

    def optional(self) -> "Optional":
        return optional(self)

    def exactly(self, count: int) -> "QuantifyExact":
        return exactly(count, self)

    def at_least(self, count: int) -> "QuantifyMinimum":
        return at_least(count, self)

    def at_most(self, count: int) -> "QuantifyMaximum":
        return at_most(count, self)

    def between(self, min_count: int, max_count: int) -> "QuantifyBetween":
        return between(min_count, max_count, self)

    def one_or_more(self) -> "KleenePlus":
        return one_or_more(self)

    def zero_or_more(self) -> "KleeneStar":
        return zero_or_more(self)


@dataclass
class String(Term):
    value: str

    def _display_node(self) -> str:
        return f"String('{self.value}')"

    def __repr__(self):
        return f"String(value='{self.value}')"


@dataclass
class Regex(Term):
    pattern: str

    def _display_node(self) -> str:
        return f"Regex('{self.pattern}')"

    def __repr__(self):
        return f"Regex(pattern='{self.pattern}')"

    def to_regex(self) -> str:
        return self.pattern


class JsonSchema(Term):
    def __init__(self, schema: Union[dict, str, type[BaseModel]]):
        if isinstance(schema, dict):
            schema_str = json.dumps(schema)
        elif isinstance(schema, str):
            schema_str = schema
        elif issubclass(schema, BaseModel):
            schema_str = json.dumps(schema.model_json_schema())
        else:
            raise ValueError(
                f"Cannot parse schema {json_schema}. The schema must be either "
                + "a Pydantic class, a dictionary or a string that contains the JSON "
                + "schema specification"
            )

        self.schema = schema_str

    def _display_node(self) -> str:
        return f"JsonSchema('{self.schema}')"

    def __repr__(self):
        return f"JsonSchema(schema='{self.schema}')"


@dataclass
class KleeneStar(Term):
    term: Term

    def _display_node(self) -> str:
        return "KleeneStar(*)"

    def _display_children(self, indent: str) -> str:
        return self.term.display_ascii_tree(indent, True)

    def __repr__(self):
        return f"KleeneStar(term={repr(self.term)})"


@dataclass
class KleenePlus(Term):
    term: Term

    def _display_node(self) -> str:
        return "KleenePlus(+)"

    def _display_children(self, indent: str) -> str:
        return self.term.display_ascii_tree(indent, True)

    def __repr__(self):
        return f"KleenePlus(term={repr(self.term)})"


@dataclass
class Optional(Term):
    term: Term

    def _display_node(self) -> str:
        return "Optional(?)"

    def _display_children(self, indent: str) -> str:
        return self.term.display_ascii_tree(indent, True)

    def __repr__(self):
        return f"Optional(term={repr(self.term)})"


@dataclass
class Alternatives(Term):
    terms: List[Term]

    def _display_node(self) -> str:
        return "Alternatives(|)"

    def _display_children(self, indent: str) -> str:
        return "".join(
            term.display_ascii_tree(indent, i == len(self.terms) - 1)
            for i, term in enumerate(self.terms)
        )

    def __repr__(self):
        return f"Alternatives(terms={repr(self.terms)})"


@dataclass
class Sequence(Term):
    terms: List[Term]

    def _display_node(self) -> str:
        return "Sequence"

    def _display_children(self, indent: str) -> str:
        return "".join(
            term.display_ascii_tree(indent, i == len(self.terms) - 1)
            for i, term in enumerate(self.terms)
        )

    def __repr__(self):
        return f"Sequence(terms={repr(self.terms)})"


@dataclass
class QuantifyExact(Term):
    term: Term
    count: int

    def _display_node(self) -> str:
        return f"Quantify({{{self.count}}})"

    def _display_children(self, indent: str) -> str:
        return self.term.display_ascii_tree(indent, True)

    def __repr__(self):
        return f"QuantifyExact(term={repr(self.term)}, count={repr(self.count)})"


@dataclass
class QuantifyMinimum(Term):
    term: Term
    min_count: int

    def _display_node(self) -> str:
        return f"Quantify({{{self.min_count},}})"

    def _display_children(self, indent: str) -> str:
        return self.term.display_ascii_tree(indent, True)

    def __repr__(self):
        return (
            f"QuantifyMinimum(term={repr(self.term)}, min_count={repr(self.min_count)})"
        )


@dataclass
class QuantifyMaximum(Term):
    term: Term
    max_count: int

    def _display_node(self) -> str:
        return f"Quantify({{,{self.max_count}}})"

    def _display_children(self, indent: str) -> str:
        return self.term.display_ascii_tree(indent, True)

    def __repr__(self):
        return (
            f"QuantifyMaximum(term={repr(self.term)}, max_count={repr(self.max_count)})"
        )


@dataclass
class QuantifyBetween(Term):
    term: Term
    min_count: int
    max_count: int

    def __post_init__(self):
        if self.min_count > self.max_count:
            raise ValueError(
                "QuantifyBetween: `max_count` must be greater than `min_count`."
            )

    def _display_node(self) -> str:
        return f"Quantify({{{self.min_count},{self.max_count}}})"

    def _display_children(self, indent: str) -> str:
        return self.term.display_ascii_tree(indent, True)

    def __repr__(self):
        return f"QuantifyBetween(term={repr(self.term)}, min_count={repr(self.min_count)}, max_count={repr(self.max_count)})"


def regex(pattern: str):
    return Regex(pattern)


def json_schema(schema: Union[str, dict, type[BaseModel]]):
    return JsonSchema(schema)


def either(*terms: Union[str, Term]):
    """Represents an alternative between different terms or strings.

    This factory function automatically translates string arguments
    into `String` objects.
    """
    terms = [String(arg) if isinstance(arg, str) else arg for arg in terms]
    return Alternatives(terms)


def optional(term: Union[Term, str]) -> Optional:
    term = String(term) if isinstance(term, str) else term
    return Optional(term)


def exactly(count: int, term: Union[Term, str]) -> QuantifyExact:
    """Repeat the term exactly `count` times."""
    term = String(term) if isinstance(term, str) else term
    return QuantifyExact(term, count)


def at_least(count: int, term: Union[Term, str]) -> QuantifyMinimum:
    """Repeat the term at least `count` times."""
    term = String(term) if isinstance(term, str) else term
    return QuantifyMinimum(term, count)


def at_most(count: int, term: Union[Term, str]) -> QuantifyMaximum:
    """Repeat the term exactly `count` times."""
    term = String(term) if isinstance(term, str) else term
    return QuantifyMaximum(term, count)


def between(min_count: int, max_count: int, term: Union[Term, str]) -> QuantifyBetween:
    term = String(term) if isinstance(term, str) else term
    return QuantifyBetween(term, min_count, max_count)


def zero_or_more(term: Union[Term, str]) -> KleeneStar:
    term = String(term) if isinstance(term, str) else term
    return KleeneStar(term)


def one_or_more(term: Union[Term, str]) -> KleenePlus:
    term = String(term) if isinstance(term, str) else term
    return KleenePlus(term)


def to_regex(term: Term) -> str:
    """Convert a term to a regular expression.

    We only consider self-contained terms that do not refer to another rule.

    """
    match term:
        case String():
            return re.escape(term.value)
        case Regex():
            return f"({term.pattern})"
        case JsonSchema():
            regex_str = build_regex_from_schema(term.schema)
            return f"({regex_str})"
        case KleeneStar():
            return f"({to_regex(term.term)})*"
        case KleenePlus():
            return f"({to_regex(term.term)})+"
        case Optional():
            return f"({to_regex(term.term)})?"
        case Alternatives():
            regexes = [to_regex(subterm) for subterm in term.terms]
            return f"({'|'.join(regexes)})"
        case Sequence():
            regexes = [to_regex(subterm) for subterm in term.terms]
            return f"{''.join(regexes)}"
        case QuantifyExact():
            return f"({to_regex(term.term)}){{{term.count}}}"
        case QuantifyMinimum():
            return f"({to_regex(term.term)}){{{term.min_count},}}"
        case QuantifyMaximum():
            return f"({to_regex(term.term)}){{,{term.max_count}}}"
        case QuantifyBetween():
            return f"({to_regex(term.term)}){{{term.min_count},{term.max_count}}}"
        case _:
            raise TypeError(
                f"Cannot convert object {repr(term)} to a regular expression."
            )
