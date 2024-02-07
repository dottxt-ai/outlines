from copy import copy, deepcopy
from typing import Dict, Optional

from lark import Lark
from lark.lexer import Token
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser_state import ParserState


class FastParserState(ParserState):
    """
    Lark ParserState with optimized copying.
    Works with Outlines because we don't perform
    any operations which mutate Tokens
    """

    copy_memo: Dict[str, Token] = {}

    def __copy__(self):
        new_value_stack = []
        for value in self.value_stack:
            key = f"{id(self)}_{id(value)}"
            if key not in self.copy_memo:
                self.copy_memo[key] = deepcopy(value, self.copy_memo)
            new_value_stack.append(self.copy_memo[key])

        new_instance = type(self)(
            self.parse_conf,
            self.lexer,
            copy(self.state_stack),
            new_value_stack,
        )

        self.copy_memo[id(self)] = new_instance
        return new_instance


class FastInteractiveParser(InteractiveParser):
    """
    InteractiveParser which uses FastParserState to manage its parse table
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser_state = FastParserState(
            self.parser_state.parse_conf,
            self.parser_state.lexer,
            self.parser_state.state_stack,
            self.parser_state.value_stack,
        )
        self.hash_val = None

    def __hash__(self):
        if self.hash_val is None:
            self.hash_val = hash(tuple(self.parser_state.state_stack))
        return self.hash_val

    def __copy__(self):
        return type(self)(
            self.parser,
            copy(self.parser_state),
            copy(self.lexer_thread),
        )


class FastLark(Lark):
    """
    Lark which uses FastInteractiveParser for interactive mode
    """

    def parse_interactive(
        self, text: Optional[str] = None, start: Optional[str] = None
    ) -> "InteractiveParser":
        base_interactive_parser = self.parser.parse_interactive(text, start=start)
        return FastInteractiveParser(
            base_interactive_parser.parser,
            base_interactive_parser.parser_state,
            base_interactive_parser.lexer_thread,
        )
