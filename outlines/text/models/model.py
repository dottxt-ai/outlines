from outlines.graph import Apply, Op
from outlines.text.var import StringVariable, as_string


class LanguageModel(Op):
    """An `Op` that produces a sample from a language model.

    The output of language models in outlines is modeled as a random variable.
    Therefore, calling a language model will return a random sequence (via
    ancestral sampling) by default. Other decoding methods are constructed
    as graph transformations.

    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def make_node(self, prompt):
        prompt = as_string(prompt)
        out = StringVariable()
        if self.name is not None:
            out.name = self.name

        return Apply(self, [prompt], [out])

    def perform(self, prompt):
        tokens = self.encode(prompt)
        sampled_tokens = self.sample(tokens)
        outputs = self.decode(sampled_tokens)
        return (outputs,)

    def sample(self, tokens):
        raise NotImplementedError

    def logprob(self, prompt, context):
        """Return the log-probability of each token in the vocabulary given the
        input prompt and the current context (previously generated tokens).

        # TODO: Implement `logprob` as a graph transformation?

        Parameters
        ----------
        prompt
            The input to the language model, parameter of the distribution.
        context
            A sequence that contains the previously generated tokens that
            are part of the context window. This sequence can be shorter
            than the total sequence generated so far if the context length
            has been reached.

        Returns
        -------
        A sequence that represents the log-probability distribution over the
        tokens.

        """
        raise NotImplementedError

    def encode(self, sequence: str):
        """Encode the given sequence.

        Defaults to a pass-through so it does not have to be implemented by
        subclasses that represent an integration to an API that take text as an
        input.

        """
        return sequence

    def decode(self, ids) -> str:
        """Decode a list of ids to a string.

        Defaults to a pass-through so it does not have to be implemented by
        subclasses that represent an integration to an API that returns text.

        """
        return ids
