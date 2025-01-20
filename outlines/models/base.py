from abc import ABC


class Model(ABC):
    """Base class for all models."""

    def __call__(self, model_input, output_type=None, **inference_kwargs):
        from outlines.generate import Generator

        return Generator(self, output_type)(model_input, **inference_kwargs)
