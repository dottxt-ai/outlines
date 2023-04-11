import numpy as np

from outlines.graph import Apply, Op, Variable
from outlines.text.var import StringVariable, as_string


class IntVariable(Variable):
    """Represents an integer value.

    This is needed to define the operations on `ListVariable`s. We should instead
    rely on Aesara for all of this.
    """


class IntConstant(IntVariable):
    """Represents an integer value.

    This is needed to define the operations on `ListVariable`s. We should instead
    rely on Aesara for all of this.
    """

    def __init__(self, value):
        self.value = value
        super().__init__()


class Choice(Op):
    def __call__(self, choices, k=1):
        return super().__call__(choices, k)

    def make_node(self, choices, k):
        var_choices = []
        for choice in choices:
            if not isinstance(choice, StringVariable):
                choice = as_string(choice)
            var_choices.append(choice)

        out = [StringVariable() for _ in range(k)]
        k = IntConstant(k)

        return Apply(self, var_choices + [k], out)

    def perform(self, inputs):
        values, k = inputs[:-1], inputs[-1]
        rng = np.random.default_rng()
        idx = list(rng.integers(len(values), size=k))
        return [values[i] for i in idx]


choice = Choice()
