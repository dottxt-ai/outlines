"""Graph objects and manipulation functions.

Manipulating prompts and operations in Outlines implicitly defines a graph that
can be explored, rewritten and compiled.

This module defines the basic types these graphs are build from:

- `Variable` nodes represent constants or results of computation;
- `Op`s represent the operations performed on variables;
- `Apply` nodes represent the application of an `Op` onto one or several
  variables.

This graph structure is a simplified version of the graph `Aesara
<https://github.com/aesara-devs/aesara>`_ uses to represents mathematical
operations on arrays. It is possible that Aesara may be used as a backend for
Outlines in the near future.

"""
from typing import Any, Iterable, List, Optional, Reversible, Sequence, Tuple, Union


class Node:
    r"""A node in an Outlines graph.

    Graphs contain two kinds of nodes: `Variable`\s and `Apply`\s. Each `Node`
    keeps track of its parents and edges are thus not represented.

    """
    name: Optional[str]

    def get_parents(self) -> List:
        """Return a list of this node's parents."""
        raise NotImplementedError()


class Variable(Node):
    """A `Variable` is a node in an expression graph that represents a variable.

    There are a few kind of `Variable` to be aware of:

    - `StringVariable` is a subclass of `Variable` that represents a ``str`` object.
    - `ImageVariable` is a subclass of `Variable` that represents image objects.

    """

    def __init__(
        self,
        owner: Optional["Apply"] = None,
        index: Optional[int] = None,
        name: Optional[str] = None,
    ):
        if owner is not None and not isinstance(owner, Apply):
            raise TypeError("owner must be an Apply instance")
        self.owner = owner

        if index is not None and not isinstance(index, int):
            raise TypeError("index must be an int")
        self.index = index

        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string")
        self.name = name

    def __str__(self):
        """Return a ``str`` representation of the `Variable`."""
        if self.name is not None:
            return self.name
        if self.owner is not None:
            op = self.owner.op
            if self.index == 0:
                return f"{str(op)}.out"
            else:
                return f"{str(op)}.{str(self.index)}"
        else:
            return f"<{getattr(type(self), '__name__')}>"


class Apply(Node):
    """An `Apply` node represents the application of an `Op` to variables.

    It is instantiated by calling the `Op.make_node` method with a list of
    inputs. The `Apply` node is in charge of filtering the inputs and outputs.

    Attribute
    ---------
    op
        The operation that produces `outputs` given `inputs`.
    inputs
        The arguments of the expression modeled by the `Apply` node.
    outputs
        The outputs of the expression modeled by the `Apply` node.

    """

    def __init__(
        self, op: "Op", inputs: Sequence["Variable"], outputs: Sequence["Variable"]
    ):
        if not isinstance(inputs, Sequence):
            raise TypeError("The inputs of an Apply node must be a sequence type")

        if not isinstance(outputs, Sequence):
            raise TypeError("The outputs of an Apply node must be a sequence type")

        self.op = op
        self.inputs: List[Variable] = []

        # Filter inputs
        for input in inputs:
            if isinstance(input, Variable):
                self.inputs.append(input)
            else:
                raise TypeError(
                    f"The 'inputs' argument to an Apply node must contain Variable instances, got {input} instead."
                )

        self.outputs: List[Variable] = []
        # Filter outputs
        for i, output in enumerate(outputs):
            if isinstance(output, Variable):
                if output.owner is None:
                    output.owner = self
                    output.index = i
                elif output.owner is not self or output.index != i:
                    raise ValueError(
                        "All outputs passed to an Apply node must belong to it."
                    )
                self.outputs.append(output)
            else:
                raise TypeError(
                    f"The 'outputs' to argument to an Apply node must contain Variable instance, got {output} instead"
                )

    def get_parents(self) -> List[Variable]:
        return list(self.inputs)


class Op:
    """Represents and constructs operations in a graph.

    An `Op` instance has the following responsibilities:

    * Construct `Apply` nodes via the :meth:`Op.make_node` method
    * Perform the computation of the modeled operation via the
      :meth:`Op.perform` method.

    A user that wants to add new capabilities to the libraries: generative
    model, API interactions, tools, etc. will need to subclass `Op` and
    implement the :meth:`Op.perform` and :meth:`Op.make_node` methods.

    """

    def make_node(self, *inputs: Variable) -> Apply:
        r"""Construct an `Apply` node that represents the application of this
        operation to the given inputs.

        This must be implemented by subclasses as it specifies the input
        and output types of the `Apply` node.

        Parameters
        ----------
        inputs
            The `Variable`\s that represent the inputs of this operation

        Returns
        -------
        The constructed `Apply` node.

        """
        raise NotImplementedError

    def __call__(self, *inputs: Variable) -> Union[Variable, List[Variable]]:
        """Calls :meth:`Op.make_node` to construct an `Apply` node."""

        node = self.make_node(*inputs)
        if len(node.outputs) == 1:
            return node.outputs[0]
        else:
            return node.outputs

    def perform(self, inputs: Tuple[Any]) -> Tuple[Any]:
        """Apply the functions to the inputs and return the output.

        Parameters
        ----------
        inputs
            Sequence of non-symbolic/numeric/text intputs.

        Returns
        -------
        The non-symbolic/numerica/text outputs of the function that this
        operation represents as a tuple.

        """
        raise NotImplementedError

    def __str__(self):
        """Return a ``str`` representation of the `Op`."""
        return getattr(type(self), "__name__")


def io_toposort(
    inputs: Iterable[Variable], outputs: Reversible[Variable]
) -> List[Apply]:
    """Sort the graph topologically starting from the inputs to the outputs.

    This function is typically used when compiling the graph, where we need
    to apply operators in the correct order to go from the user inputs to
    the program outputs.

    Parameters
    ----------
    inputs
        Graph inputs.
    outputs
        Graph outputs.

    """
    computed = set(inputs)
    todo = [o.owner for o in reversed(outputs) if o.owner]
    order = []
    while todo:
        node = todo.pop()
        if node.outputs[0] in computed:
            continue
        if all(i in computed or i.owner is None for i in node.inputs):
            computed.update(node.outputs)
            order.append(node)
        else:
            todo.append(node)
            todo.extend(i.owner for i in node.inputs if i.owner)

    return order
