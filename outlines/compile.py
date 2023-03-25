from typing import Callable, Iterable, Reversible

from outlines.graph import Variable, io_toposort
from outlines.text.var import StringConstant


def compile(inputs: Iterable[Variable], outputs: Reversible[Variable]) -> Callable:
    r"""Compile an Outlines graph into an executable function.

    `compile` first sorts the graph defined by the input and output nodes
    topologically. It then visits the nodes one by one and executes their
    `Op`'s `perform` method, fetching and storing their values in a map.

    Parameters
    ----------
    inputs
        The symbolic `Variable`\s that represent the inputs of the compiled
        program.
    outputs
        The symbolic `Variable`\s that represent the outputs of the compiled
        program.

    Returns
    -------
    A function which returns the values of the output nodes when passed the values
    of the input nodes as arguments.

    """
    sorted = io_toposort(inputs, outputs)

    def fn(*values):
        storage_map = {s: v for s, v in zip(inputs, values)}

        for node in sorted:
            for i in node.inputs:
                if isinstance(i, StringConstant):
                    storage_map[i] = i.value
            node_inputs = [storage_map[i] for i in node.inputs]
            results = node.op.perform(*node_inputs)
            for i, o in enumerate(node.outputs):
                storage_map[o] = results[i]

        if len(outputs) == 1:
            return storage_map[outputs[0]]
        else:
            return tuple(storage_map[o] for o in outputs)

    return fn
