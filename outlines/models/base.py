from abc import ABC, abstractmethod


class ModelFormatter(ABC):
    """Base class for all model formatters."""

    @abstractmethod
    def format_input(self, model_input):
        ...

    @abstractmethod
    def format_output_type(self, output_type):
        ...


class Model(ABC):
    """Base class for all models."""

    formatter: ModelFormatter

    def __call__(self, model_input, output_type=None, **inference_kwargs):
        from outlines.generate import Generator

        return Generator(self, output_type)(model_input, **inference_kwargs)
