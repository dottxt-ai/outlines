"""Functionalities to wrap user-defined functions as Ops.

The content of this module is heavily inspired by the design of
`Aesara <https://github.com/aesara-devs/aesara>`_.

"""
import inspect
from typing import Callable, Sequence, Type

from outlines.graph import Op, Variable
from outlines.text.var import StringVariable


class FromFunctionOp(Op):
    """Build an outlines Op around a function."""

    def __init__(
        self,
        fn: Callable,
        input_types: Sequence[Type[Variable]],
        output_types: Sequence[Type[Variable]],
    ):
        self._fn = fn
        self.input_types = input_types
        self.output_types = output_types

    def __str__(self):
        return f"FromFunctionOp({self._fn.__name__})"

    def perform(self, *inputs):
        outs = self._fn(*inputs)
        if not isinstance(outs, (list, tuple)):
            outs = (outs,)

        return outs


def fn(function: Callable):
    """Decorator that converts a Python function into an Outlines `Op`
    that will call the function as its implementation.

    The user must specify the types of the inputs and outputs as type
    hints.

    """
    sig = inspect.signature(function)

    inputs = []
    for name, parameter in sig.parameters.items():
        if parameter.annotation == str:
            inputs.append(StringVariable)
        elif parameter.annotation == inspect._empty:
            raise TypeError(
                "You need to specify the function's input types as type hints."
            )
        else:
            raise TypeError(
                "The `fn` decorator currently only supports string arguments."
            )

    outputs = []
    if sig.return_annotation == str:
        outputs.append(StringVariable)
    elif sig.return_annotation == inspect._empty:
        raise TypeError(
            "You need to specify the function's output types as type hints."
        )
    else:
        raise TypeError(
            "The `fn` decorator currently only supports string return types"
        )

    return FromFunctionOp(function, input_types=inputs, output_types=outputs)
