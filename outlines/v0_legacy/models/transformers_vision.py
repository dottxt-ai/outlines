### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

from typing import TYPE_CHECKING, Optional
import warnings

from outlines.models.transformers import (
    Transformers,
    TransformersMultiModalTypeAdapter,
)


class TransformersVision(Transformers):
    def __init__(self, model, tokenizer, processor):
        self.processor = processor
        self.processor.padding_side = "left"
        self.processor.pad_token = "[PAD]"

        super().__init__(model, tokenizer)

        self.type_adapter = TransformersMultiModalTypeAdapter()


    def _prepare_model_inputs(self, model_input, output_type):
        model_input = self.type_adapter.format_input(model_input)
        inputs = self.processor(
            **model_input, padding=True, return_tensors="pt"
        ).to(self.model.device)

        return model_input["text"], inputs


def transformers_vision(
    model_name: str,
    model_class,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    processor_kwargs: dict = {},
    tokenizer_class=None,
    processor_class=None,
):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    model_class
        The `PreTrainedModel` class from transformers to use in initializing the vision model from `model_name`.
        https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel
    device
        The device(s) on which the model should be loaded. This overrides
        the `device_map` entry in `model_kwargs` when provided.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the model.
    processor_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the processor.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    warnings.warn("""
        The `transformers_vision` function is deprecated starting from v1.0.0.
        Do not use it. Support for it will be removed in v1.1.0.
        Instead, you should instantiate a `TransformersMultiModal` model with
        the `outlines.from_transformers` function that takes a transformers
        model and a processor as arguments. For example:
        ```python
        from transformers import AutoModelForCausalLM, AutoProcessor
        from outlines import from_transformers
        model = from_transformers(
            AutoModelForCausalLM.from_pretrained("<model_name>"),
            AutoProcessor.from_pretrained("<model_name>")
        )
        ```
        """,
        DeprecationWarning,
        stacklevel=2,
    )

    if processor_class is None or tokenizer_class is None:
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError:  # pragma: no cover
            raise ImportError(
                "The `transformers` library needs to be installed in order to use `transformers` models."
            )
    if processor_class is None:
        processor_class = AutoProcessor
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    if device is not None:
        model_kwargs["device_map"] = device

    model = model_class.from_pretrained(model_name, **model_kwargs)

    processor_kwargs.setdefault("padding_side", "left")
    processor_kwargs.setdefault("pad_token", "[PAD]")
    processor = processor_class.from_pretrained(model_name, **processor_kwargs)
    tokenizer = tokenizer_class.from_pretrained(model_name, **processor_kwargs)

    return TransformersVision(model, tokenizer, processor)
