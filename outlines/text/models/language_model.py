from outlines.graph import Apply, Op
from outlines.text.var import StringVariable, as_string


class LanguageModel(Op):
    """An `Op` that produces a sample from a language model.

    The output of language models in outlines is represented as a random
    variable.  Therefore, calling a language model will return a random sequence
    (via ancestral sampling) by default. Other decoding methods are constructed
    as graph transformations.

    """

    def __init__(self, name=None):
        """Instantiate the `LanguageModel` `Op`.

        Parameters
        ----------
        name
            The name of the `Op` in the graph.

        """
        super().__init__()
        self.name = name

    def __call__(self, prompt, stops_at=None, name=None):
        """Create the `Apply` node that represents the `Op`'s application to inputs.

        Parameters
        ----------
        prompt
            The prompt used to condition the language model's sampling procedure.
        name
            The name of the output variable in the graph.

        """
        res = super().__call__(prompt)

        if name is not None:
            res.name = name

        return res

    def make_node(self, prompt):
        prompt = as_string(prompt)
        out = StringVariable()

        return Apply(self, [prompt], [out])

    def perform(self, prompt):
        return NotImplementedError
