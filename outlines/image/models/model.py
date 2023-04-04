from typing import Tuple

from PIL.Image import Image as PILImage

from outlines.graph import Apply, Op, Variable
from outlines.image.var import ImageVariable
from outlines.text.var import as_string


class ImageModel(Op):
    """An `Op` that produces a sample image from generative model.

    The output of generative models in outlines is modeled as a random variable.
    Therefore, calling an image generative model will return a random image by
    default.

    Attributes
    ----------
    name
        The `Op`'s name in the graph.

    """

    def __init__(self, name=None):
        """Instantiate the `ImageModel` `Op`.

        Parameters
        ----------
        name
            The name of the `Op` in the graph.

        """
        super().__init__()
        self.name = name

    def __call__(self, prompt, name=None):
        """Create the `Apply` node that represents the `Op`'s application to inputs.

        Parameters
        ----------
        prompt
            The prompt used to condition the generative model's sampling procedure.
        name
            The name of the output variable in the graph.

        """
        res = super().__call__(prompt)

        if name is not None:
            res.name = name

        return res

    def make_node(self, prompt: Variable) -> Apply:  # type: ignore
        prompt = as_string(prompt)
        out = ImageVariable()

        return Apply(self, [prompt], [out])

    def perform(self, prompt: str) -> Tuple[PILImage]:  # type: ignore
        """Perform the operations represented by this `Op` on the input prompt.

        This defaults to sampling a new image. Other decoding methods act by
        patching this method.

        """
        return (self.sample(prompt),)

    def sample(self, prompt: str) -> PILImage:
        """Sample a new image given the input prompt."""
        raise NotImplementedError
