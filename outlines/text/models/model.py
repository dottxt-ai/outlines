from outlines.graph import Apply, Op
from outlines.text.var import StringVariable, as_string


class LanguageModel(Op):
    def make_node(self, prompt):
        prompt = as_string(prompt)
        out = StringVariable()
        return Apply(self, [prompt], [out])

    def perform(self, prompt):
        return self.sample(prompt)

    def sample(self, prompt):
        return (f"2x{prompt}",)

    def logprob(self, prompt, context):
        """Return the log-probability of each token in the vocabulary given the
        input prompt and the current context (previously generated tokens).

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
