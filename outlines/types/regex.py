import re
from typing import Any, Union
import typing
import json

from outlines_core.fsm.json_schema import build_regex_from_schema

from .dsl import String, Regex, KleeneStar, KleenePlus, Optional, Alternatives, Sequence, QuantifyExact, QuantifyBetween, QuantifyMinimum, QuantifyMaximum, Term, JsonSchema


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
