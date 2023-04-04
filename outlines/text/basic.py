"""Basic `StringVariable` manipulations."""
import outlines
from outlines.graph import Apply, Op
from outlines.text.var import StringVariable

__all__ = ["add"]


class Add(Op):
    def make_node(self, s, t):
        s = outlines.text.as_string(s)
        t = outlines.text.as_string(t)
        out = StringVariable()
        return Apply(self, [s, t], [out])

    def perform(self, s, t):
        return (s + t,)


add = Add()
