from functools import singledispatchmethod
from typing import Dict, Union

from mako import lexer
from mako.parsetree import Expression, Text

from outlines.graph import Op
from outlines.text.var import StringVariable, as_string


class Script:
    """Represents a scripted interaction with generative models.

    The `Script` class provides a convenient way to define Outlines graph using
    the Mako templating languages.`Scripts` are instantiated by passing a string
    that represents the flow of interaction with one or several generative models.

    """

    def __init__(self, script):
        self.parsetree = lexer.Lexer(script).parse()

    def __call__(self, **inputs: Dict[str, Union[StringVariable, Op]]):
        nodes = self.parsetree.nodes
        graph = self.parse_node(nodes[0], inputs)
        for node in self.parsetree.nodes[1:]:
            graph = graph + self.parse_node(node, inputs)

        return graph

    @singledispatchmethod
    def parse_node(self, node, inputs):
        raise NotImplementedError(f"Cannot transpile {node} to an Outlines graph.")

    @parse_node.register(Text)
    def parse_Text(self, node, inputs):
        return as_string(node.content)

    @parse_node.register(Expression)
    def parse_Expression(self, node, inputs):
        try:
            return as_string(inputs[node.text])
        except KeyError:
            raise TypeError(
                f"Prompt evaluation missing 1 required argument: '{node.text}'"
            )


script = Script
