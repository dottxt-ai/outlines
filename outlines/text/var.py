from functools import singledispatch

import outlines.text as ot
from outlines.graph import Variable


class StringVariable(Variable):
    """Subclass to add the string operators to `Variable`."""

    def __init__(self, owner=None, index=None, name=None):
        super().__init__(owner, index, name)

    def __add__(self, other):
        return ot.add(self, other)

    def __radd__(self, other):
        return ot.add(other, self)


string = StringVariable


class StringConstant(StringVariable):
    """Constant `StringVariable` that corresponds to user input."""

    def __init__(self, value, name=None):
        self.value = value
        super().__init__(name=name)

    def __str__(self):
        if self.name is not None:
            name = self.name
        else:
            name = "StringConstant"
        return f"{name}{{'{self.value}'}}"


@singledispatch
def as_string(x, name=None):
    """Convert `x` into an equivalent `StringVariable`.

    This function can be used to turn `str`, `int` and `float` instances into a
    `StringVariable`.

    Parameters
    ----------
    x
        The object that will we converted into a `StringVariable`.
    name
        If a new `StringVariable` instance is created it will be attributed this
        name.

    """
    raise TypeError(f"{x} cannot be cast into a string")


@as_string.register(str)
def as_string_variable_strings(x, name=None):
    return StringConstant(x, name)


@as_string.register(int)
@as_string.register(float)
def as_string_variable_numbers(x, name=None):
    return StringConstant(str(x), name)


@as_string.register(StringVariable)
def as_string_variable_StringVariable(x, name=None):
    return x
