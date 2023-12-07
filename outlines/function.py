from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

from outlines import generate, models

if TYPE_CHECKING:
    from outlines.generate.api import SequenceGenerator


@dataclass
class Function:
    """Represents an Outlines function.

    Functions are a convenient way to encapsulate a prompt template, a language
    model and a Pydantic model that define the output structure. Once defined,
    the function can be called with arguments that will be used to render the
    prompt template.

    """

    prompt_template: Callable
    model_name: str
    schema: Union[str, Callable, object]
    generator: Optional["SequenceGenerator"] = None

    def init_generator(self):
        """Load the model and initialize the generator."""
        model = models.transformers(self.model_name)
        self.generator = generate.json(model, self.schema)

    def __call__(self, *args, **kwargs):
        """Call the function.

        .. warning::

           This currently does not support batching.

        Parameters
        ----------
        args
            Values to pass to the prompt template as positional arguments.
        kwargs
            Values to pass to the prompt template as keyword arguments.

        """
        if self.generator is None:
            self.init_generator()

        prompt = self.prompt_template(*args, **kwargs)
        return self.generator(prompt)
