from functools import singledispatchmethod

from outlines.models import Transformers
from outlines.models.base import ModelTypeAdapter


class TransformersVisionTypeAdapter(ModelTypeAdapter):
    """Type adapter for TransformersVision models."""

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the prompt argument to pass to the model.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        raise NotImplementedError(
            f"The input type {input} is not available. Please provide a "
            "dictionary with the following format: "
            "{'prompts': Union[str, List[str]], 'images': Union[Any, List[Any]]}"
            "Make sure the number of image tags in the prompts is equal to the "
            "number of images provided."
        )

    @format_input.register(dict)
    def format_list_input(self, model_input):
        if "prompts" not in model_input or "images" not in model_input:
            raise ValueError(
                "The input must contain the following keys: 'prompts' and 'images'."
            )
        return model_input["prompts"], model_input["images"]

    def format_output_type(self, output_type):
        """Generate the logits processor argument to pass to the model.

        Argument
        --------
        output_type
            The logits processor provided.

        """
        from transformers import LogitsProcessorList

        if output_type is not None:
            return LogitsProcessorList([output_type])
        return None


class TransformersVision(Transformers):
    """Represents a `transformers` model with a vision processor."""

    def __init__(
        self,
        model_name: str,
        model_class,
        model_kwargs: dict = {},
        tokenizer_class=None,
        tokenizer_kwargs: dict = {},
        processor_class=None,
        processor_kwargs: dict = {},
    ):
        """Create a TransformersVision model instance

        We rely on the `__init__` method of the `Transformers` class to handle
        most of the initialization and then add elements specific to vision
        models.

        Parameters
        ----------
        model_name
            The name of the transformers model to use;
        model_class
            The Transformers model class from which to create the model.
        model_kwargs
            A dictionary of keyword arguments to pass to the `from_pretrained`
            method of the model class.
        tokenizer_class
            The Transformers tokenizer class from which to create the tokenizer.
            If not provided,`AutoTokenizer` will be used.
            If you gave the name of a model that is not compatible with `AutoTokenizer`,
            you must provide a value for this parameter.
        tokenizer_kwargs
            A dictionary of keyword arguments to pass to the `from_pretrained`
            method of the tokenizer class.
        processor_class
            The Transformers processor class from which to create the processor.
            If not provided,`AutoProcessor` will be used.
            If you gave the name of a model that is not compatible with `AutoProcessor`,
            you must provide a value for this parameter.
        processor_kwargs
            A dictionary of keyword arguments to pass to the `from_pretrained`
            method of the processor class.

        """
        if processor_class is None:
            try:
                from transformers import AutoProcessor

                processor_class = AutoProcessor
            except ImportError:
                raise ImportError(
                    "The `transformers` library needs to be installed in order to use `transformers` models."
                )

        processor_kwargs.setdefault("padding_side", "left")
        processor_kwargs.setdefault("pad_token", "[PAD]")
        self.processor = processor_class.from_pretrained(model_name, **processor_kwargs)

        if tokenizer_class is None and getattr(self.processor, "tokenizer", None):
            tokenizer_class = type(self.processor.tokenizer)

        super().__init__(
            model_name,
            model_class,
            model_kwargs,
            tokenizer_class,
            tokenizer_kwargs,
        )

        self.type_adapter = TransformersVisionTypeAdapter()

    def generate(self, model_input, output_type, **inference_kwargs):
        prompts, images = self.type_adapter.format_input(model_input)

        inputs = self.processor(
            text=prompts, images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        logits_processor = self.type_adapter.format_output_type(output_type)

        generated_ids = self._generate_output_seq(
            prompts, inputs, logits_processor=logits_processor, **inference_kwargs
        )

        # if single str input, convert to a 1D outputt
        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def stream(self, model_input, output_type, **inference_kwargs):
        raise NotImplementedError(
            "Streaming is not implemented for TransformersVision models."
        )
