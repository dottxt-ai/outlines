"""Regular expression DSL and output types for structured generation.

This module contains elements related to three logical steps in the use of
output types for structured generation:

1. Definition of `Term` classes that contain output type definitions. That
   includes both terms intended to be used by themselves such as `JsonSchema`
   or `CFG` and terms that are part of the regular expression DSL such as
   `Alternatives` or `KleeneStar` (and the related functions).
2. Conversion of Python types into `Term` instances (`python_types_to_terms`).
3. Conversion of a `Term` instance into a regular expression (`to_regex`).

"""

import json
import re
import sys
import warnings
from dataclasses import dataclass
from enum import EnumMeta
from types import FunctionType
from typing import (
    Any,
    List,
    Optional as OptionalType,
    Union,
    get_args,
)
import interegular
import jsonschema
from genson import SchemaBuilder
from outlines_core.fsm.json_schema import build_regex_from_schema
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    TypeAdapter,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema as cs

import outlines.types as types
from outlines import grammars
from outlines.types.utils import (
    get_schema_from_signature,
    is_int,
    is_int_instance,
    is_float,
    is_float_instance,
    is_str,
    is_str_instance,
    is_bool,
    is_datetime,
    is_date,
    is_time,
    is_native_dict,
    is_dict_instance,
    is_dataclass,
    is_typed_dict,
    is_pydantic_model,
    is_genson_schema_builder,
    is_literal,
    is_union,
    is_enum,
    is_callable,
    is_typing_list,
    is_typing_tuple,
    is_typing_dict,
)

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import _TypedDictMeta  # type: ignore
else:  # pragma: no cover
    from typing_extensions import _TypedDictMeta  # type: ignore


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

    def __add__(self: "Term", other: "Term") -> "Sequence":
        if is_str_instance(other):
            other = String(str(other))

        return Sequence([self, other])

    def __radd__(self: "Term", other: "Term") -> "Sequence":
        if is_str_instance(other):
            other = String(str(other))

        return Sequence([other, self])

    def __or__(self: "Term", other: "Term") -> "Alternatives":
        if is_str_instance(other):
            other = String(str(other))

        return Alternatives([self, other])

    def __ror__(self: "Term", other: "Term") -> "Alternatives":
        if is_str_instance(other):
            other = String(str(other))

        return Alternatives([other, self])

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
    """Class representing a regular expression.

    Parameters
    ----------
    pattern
        The regular expression as a string.

    """
    pattern: str

    def _display_node(self) -> str:
        return f"Regex('{self.pattern}')"

    def __repr__(self):
        return f"Regex(pattern='{self.pattern}')"


@dataclass
class CFG(Term):
    """Class representing a context-free grammar.

    Parameters
    ----------
    definition
        The definition of the context-free grammar as a string.

    """
    definition: str

    def _display_node(self) -> str:
        return f"CFG('{self.definition}')"

    def __repr__(self):
        return f"CFG(definition='{self.definition}')"

    def __eq__(self, other):
        if not isinstance(other, CFG):
            return False
        return self.definition == other.definition

    @classmethod
    def from_file(cls, path: str) -> "CFG":
        """Create a CFG instance from a file containing a CFG definition.

        Parameters
        ----------
        path : str
            The path to the file containing the CFG definition.
        Returns
        -------
        CFG
            A CFG instance.

        """
        with open(path, "r") as f:
            definition = f.read()
        return cls(definition)


@dataclass
class FSM(Term):
    """Class representing a finite state machine.

    Parameters
    ----------
    fsm
        The finite state machine to store. This object must be an instance of
        `interegular.fsm.FSM`.

    """
    fsm: interegular.fsm.FSM

    def _display_node(self) -> str:
        return f"FSM({self.fsm.__repr__()})"

    def __repr__(self):
        return f"FSM(fsm={self.fsm.__repr__()})"


class JsonSchema(Term):
    """Class representing a JSON schema.

    The JSON schema object from which to instantiate the class can be a
    dictionary, a string, a Pydantic model, a typed dict, a dataclass, or a
    genSON schema builder.

    """
    def __init__(
        self,
        schema: Union[
            dict, str, type[BaseModel], _TypedDictMeta, type, SchemaBuilder
        ],
        whitespace_pattern: OptionalType[str] = None,
        ensure_ascii: bool = True,
    ):
        """
        Parameters
        ----------
        schema
            The object containing the JSON schema.
        whitespace_pattern
            The pattern to use to match whitespace characters.
        ensure_ascii
            Whether to ensure the schema is ASCII-only.

        """
        schema_str: str

        if is_dict_instance(schema):
            schema_str = json.dumps(schema, ensure_ascii=ensure_ascii)
        elif is_str_instance(schema):
            schema_str = str(schema)
        elif is_pydantic_model(schema):
            schema_str = json.dumps(schema.model_json_schema(), ensure_ascii=ensure_ascii) # type: ignore
        elif is_typed_dict(schema):
            schema_str = json.dumps(TypeAdapter(schema).json_schema(), ensure_ascii=ensure_ascii)
        elif is_dataclass(schema):
            schema_str = json.dumps(TypeAdapter(schema).json_schema(), ensure_ascii=ensure_ascii)
        elif is_genson_schema_builder(schema):
            schema_str = schema.to_json(ensure_ascii=ensure_ascii)  # type: ignore
        else:
            raise ValueError(
                f"Cannot parse schema {schema}. The schema must be either "
                + "a Pydantic class, typed dict, a dataclass, a genSON schema "
                + "builder or a string or dict that contains the JSON schema "
                + "specification"
            )

        self.schema = schema_str
        self.whitespace_pattern = whitespace_pattern

    def __post_init__(self):
        jsonschema.Draft7Validator.check_schema(json.loads(self.schema))

    def _display_node(self) -> str:
        return f"JsonSchema('{self.schema}')"

    def __repr__(self):
        return f"JsonSchema(schema='{self.schema}')"

    def __eq__(self, other):
        if not isinstance(other, JsonSchema):
            return False
        try:
            self_dict = json.loads(self.schema)
            other_dict = json.loads(other.schema)
            return self_dict == other_dict
        except json.JSONDecodeError:  # pragma: no cover
            return self.schema == other.schema

    @classmethod
    def from_file(cls, path: str) -> "JsonSchema":
        """Create a JsonSchema instance from a .json file containing a JSON
        schema.

        Parameters
        ----------
        path:
            The path to the file containing the JSON schema.
        Returns
        -------
        JsonSchema
            A JsonSchema instance.

        """
        with open(path, "r") as f:
            schema = json.load(f)
        return cls(schema)


@dataclass
class Choice(Term):
    """Class representing a choice between different items.

    Parameters
    ----------
    items
        The items to choose from.

    """
    items: List[Any]

    def _display_node(self) -> str:
        return f"Choice({repr(self.items)})"

    def __repr__(self):
        return f"Choice(items={repr(self.items)})"


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


def cfg(definition: str):
    return CFG(definition)


def fsm(fsm: interegular.fsm.FSM):
    return FSM(fsm)


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


def python_types_to_terms(ptype: Any, recursion_depth: int = 0) -> Term:
    """Convert Python types to Outlines DSL terms that constrain LLM output.

    Parameters
    ----------
    ptype
        The Python type to convert
    recursion_depth
        Current recursion depth to prevent infinite recursion

    Returns
    -------
    Term
        The corresponding DSL `Term` instance.

    """
    if recursion_depth > 10:
        raise RecursionError(
            f"Maximum recursion depth exceeded when converting {ptype}. "
            "This might be due to a recursive type definition."
        )

    # First handle Term instances
    if isinstance(ptype, Term):
        return ptype

    # Basic types
    if is_int(ptype):
        return types.integer
    elif is_float(ptype):
        return types.number
    elif is_bool(ptype):
        return types.boolean
    elif is_str(ptype):
        return types.string
    elif is_native_dict(ptype):
        return CFG(grammars.json)
    elif is_time(ptype):
        return types.time
    elif is_date(ptype):
        return types.date
    elif is_datetime(ptype):
        return types.datetime

    # Basic type instances
    if is_str_instance(ptype):
        return String(ptype)
    elif is_int_instance(ptype) or is_float_instance(ptype):
        return Regex(str(ptype))

    # Structured types
    structured_type_checks = [
        lambda x: is_dataclass(x),
        lambda x: is_typed_dict(x),
        lambda x: is_pydantic_model(x),
    ]
    if any(check(ptype) for check in structured_type_checks):
        schema = TypeAdapter(ptype).json_schema()
        return JsonSchema(schema)

    elif is_genson_schema_builder(ptype):
        schema = ptype.to_json()
        return JsonSchema(schema)

    if is_enum(ptype):
        return Alternatives(
            [
                python_types_to_terms(member, recursion_depth + 1)
                for member in _get_enum_members(ptype)
            ]
        )

    args = get_args(ptype)
    if is_literal(ptype):
        return _handle_literal(args)
    elif is_union(ptype):
        return _handle_union(args, recursion_depth)
    elif is_typing_list(ptype):
        return _handle_list(args, recursion_depth)
    elif is_typing_tuple(ptype):
        return _handle_tuple(args, recursion_depth)
    elif is_typing_dict(ptype):
        return _handle_dict(args, recursion_depth)

    if is_callable(ptype):
        return JsonSchema(get_schema_from_signature(ptype))

    type_name = getattr(ptype, "__name__", ptype)
    raise TypeError(
        f"Type {type_name} is currently not supported. Please open an issue: "
        "https://github.com/dottxt-ai/outlines/issues"
    )


def _get_enum_members(ptype: EnumMeta) -> List[Any]:
    regular_members = [member.value for member in ptype]  # type: ignore
    function_members = []
    for key, value in ptype.__dict__.items():
        if (
            isinstance(value, FunctionType)
            and not (key.startswith('__') and key.endswith('__'))
            and key != '_generate_next_value_'  # Skip this specific method that causes issues
        ):
            function_members.append(value)
    return regular_members + function_members


def _handle_literal(args: tuple) -> Alternatives:
    return Alternatives([python_types_to_terms(arg) for arg in args])


def _handle_union(args: tuple, recursion_depth: int) -> Alternatives:
    # Handle the Optional[T] type
    if len(args) == 2 and (type(None) in args or None in args):
        other_ptype = next(arg for arg in args if arg not in (type(None), None))
        return Alternatives(
            [
                python_types_to_terms(other_ptype, recursion_depth + 1),
                String("None"),
            ]
        )
    return Alternatives(
        [python_types_to_terms(arg, recursion_depth + 1) for arg in args]
    )


def _handle_list(args: tuple, recursion_depth: int) -> Sequence:
    if args is None or len(args) > 1:
        raise TypeError(
            f"Only homogeneous lists are supported. Got multiple type arguments {args}."
        )
    item_type = python_types_to_terms(args[0], recursion_depth + 1)
    return Sequence(
        [
            String("["),
            item_type,
            KleeneStar(Sequence([String(", "), item_type])),
            String("]"),
        ]
    )


def _handle_tuple(args: tuple, recursion_depth: int) -> Union[Sequence, String]:
    if len(args) == 0 or args == ((),):
        return String("()")
    elif len(args) == 2 and args[1] is Ellipsis:
        item_term = python_types_to_terms(args[0], recursion_depth + 1)
        return Sequence(
            [
                String("("),
                item_term,
                KleeneStar(Sequence([String(", "), item_term])),
                String(")"),
            ]
        )
    else:
        items = [python_types_to_terms(arg, recursion_depth + 1) for arg in args]
        separator = String(", ")
        elements = []
        for i, item in enumerate(items):
            elements.append(item)
            if i < len(items) - 1:
                elements.append(separator)
        return Sequence([String("("), *elements, String(")")])


def _handle_dict(args: tuple, recursion_depth: int) -> Sequence:
    if args is None or len(args) != 2:
        raise TypeError(f"Dict must have exactly two type arguments. Got {args}.")
    # Add dict support with key:value pairs
    key_type = python_types_to_terms(args[0], recursion_depth + 1)
    value_type = python_types_to_terms(args[1], recursion_depth + 1)
    return Sequence(
        [
            String("{"),
            Optional(
                Sequence(
                    [
                        key_type,
                        String(":"),
                        value_type,
                        KleeneStar(
                            Sequence([String(", "), key_type, String(":"), value_type])
                        ),
                    ]
                )
            ),
            String("}"),
        ]
    )


def to_regex(term: Term) -> str:
    """Convert a term to a regular expression.

    We only consider self-contained terms that do not refer to another rule.

    Parameters
    ----------
    term
        The term to convert to a regular expression.

    Returns
    -------
    str
        The regular expression as a string.

    """
    match term:
        case String():
            return re.escape(term.value)
        case Regex():
            return f"({term.pattern})"
        case JsonSchema():
            regex_str = build_regex_from_schema(term.schema, term.whitespace_pattern)
            return f"({regex_str})"
        case Choice():
            regexes = [to_regex(python_types_to_terms(item)) for item in term.items]
            return f"({'|'.join(regexes)})"
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
