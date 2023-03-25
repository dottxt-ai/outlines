from functools import singledispatchmethod
from typing import Dict, Union

from mako import lexer
from mako.parsetree import Expression, Text

from outlines.graph import Op
from outlines.text.models import LanguageModel
from outlines.text.var import StringVariable, as_string


class Script:
    """Represents a scripted interaction with generative models.

    The `Script` class provides a convenient way to define Outlines graph using
    the Mako templating languages.`Scripts` are instantiated by passing a string
    that represents the flow of interaction with one or several generative models.

    """

    def __init__(self, script):
        self.parsetree = lexer.Lexer(script).parse()
        self.model_outputs = {}

    def __call__(self, **inputs: Dict[str, Union[StringVariable, Op]]):
        """Create an Outlines graph from a Mako template.

        When one calls a `Script` instance with arguments that represent
        variables in the template, Outlines parses the template and iteratively
        builds the graph it represents before returning it.

        """
        nodes = self.parsetree.nodes
        graph = self.parse_node(nodes[0], inputs, "")
        for node in self.parsetree.nodes[1:]:
            graph = graph + self.parse_node(node, inputs, graph)

        return graph

    @singledispatchmethod
    def parse_node(self, node, inputs, graph):
        raise NotImplementedError(f"Cannot transpile {node} to an Outlines graph.")

    @parse_node.register(Text)
    def parse_Text(self, node, inputs, graph):
        """Parse Mako's `Text` nodes.

        `Text` nodes corresponds to `StringConstants` in Outline's language.

        """
        return as_string(node.content)

    @parse_node.register(Expression)
    def parse_Expression(self, node, inputs, graph):
        """Parse Mako's `Expression` nodes.

        We first fetch the argument that the user passed to the `__call__`
        method that corresponds to the current variable name. Then we check if
        this argument has already been seen; if that's the case we assume the
        user is referencing the output of a previously-run LM and add the
        corresponding node.

        """
        try:
            user_input = inputs[node.text]
            if isinstance(user_input, LanguageModel):
                try:
                    return self.model_outputs[node.text]
                except KeyError:
                    output = user_input(graph)
                    self.model_outputs[node.text] = output
                    return output
            else:
                return as_string(inputs[node.text])
        except KeyError:
            raise TypeError(
                f"Prompt evaluation missing 1 required argument: '{node.text}'"
            )


script = Script
