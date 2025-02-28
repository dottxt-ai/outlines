from abc import ABC, abstractmethod


class ModelTypeAdapter(ABC):
    """Base class for all model type adapters.

    A type adapter instance must be given as a value to the `type_adapter`
    attribute when instantiating a model.
    The type adapter is responsible for formatting the input and output types
    passed to the model to match the specific format expected by the
    associated model.

    """

    @abstractmethod
    def format_input(self, model_input):
        """Format the user input to the expected format of the model.

        For API-based models, it typically means creating the `messages`
        argument passed to the client. For local models, it can mean casting
        the input from str to list for instance.
        This method is also used to validate that the input type provided by
        the user is supported by the model.

        """
        ...

    @abstractmethod
    def format_output_type(self, output_type):
        """Format the output type to the expected format of the model.

        For API-based models, this typically means creating a `response_format`
        argument. For local models, it means formatting the logits processor to
        create the object type expected by the model.

        """
        ...


class Model(ABC):
    """Base class for all models.

    This class defines a shared `__call__` method that can be used to call the
    model directly.
    All models inheriting from this class must define a `type_adapter`
    attribute of type `ModelTypeAdapter`. The methods of the `type_adapter`
    attribute are used in the `generate` method to format the input and output
    types received by the model.

    """

    type_adapter: ModelTypeAdapter

    def __call__(self, model_input, output_type=None, **inference_kwargs):
        """Call the model.

        Users can call the model directly, in which case we will create a
        generator instance with the output type provided and call it.
        Thus, those commands are equivalent:
        ```python
        generator = Generator(model, Foo)
        generator("prompt")
        ```
        and
        ```python
        model("prompt", Foo)
        ```

        """
        from outlines import Generator

        return Generator(self, output_type)(model_input, **inference_kwargs)

    @abstractmethod
    def generate(self, model_input, output_type=None, **inference_kwargs):
        """Generate a response from the model.

        The output_type argument contains a logits processor for local models
        while it contains a type (Json, Enum...) for the API-based models.

        """
        ...
