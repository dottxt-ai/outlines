from functools import singledispatchmethod
from typing import TYPE_CHECKING

from outlines.models.base import Model, ModelTypeAdapter
from outlines.models.transformers import TransformerTokenizer

if TYPE_CHECKING:
    import mlx.nn as nn
    from transformers import PreTrainedTokenizer


__all__ = ["MLXLM", "from_mlxlm"]


class MLXLMTypeAdapter(ModelTypeAdapter):
    """Type adapter for `mlx_lm` models."""

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the prompt argument to pass to the model.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        raise NotImplementedError(
            f"The input type {input} is not available. "
            "The `mlx_lm` library does not support batch inference."
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str):
        return model_input

    def format_output_type(self, output_type):
        """Generate the logits processor argument to pass to the model.

        Argument
        --------
        output_type
            The logits processor provided.

        """
        if not output_type:
            return None
        return [output_type]


class MLXLM(Model):
    """Represents an `mlx_lm` model."""
    tensor_library_name = "mlx"

    def __init__(
        self,
        model: "nn.Module",
        tokenizer: "PreTrainedTokenizer",
    ):
        """Create a MLXLM model instance.

        Arguments
        ---------
        model
            An instance of an mlx-lm model.
        tokenizer
            An instance of an mlx-lm tokenizer or of a compatible
            transformers tokenizer.

        """
        self.model = model
        # self.mlx_tokenizer is used by the mlx-lm in its generate function
        self.mlx_tokenizer = tokenizer
        # self.tokenizer is used by the logits processor
        self.tokenizer = TransformerTokenizer(tokenizer._tokenizer)
        self.type_adapter = MLXLMTypeAdapter()

    def generate(self, model_input, output_type=None, **kwargs):
        """Generate text using `mlx-lm`.

        Arguments
        ---------
        model_input
            The prompt passed by the user.
        output_type
            The logits processor provided.
        kwargs
            The inference kwargs that can be passed to the `mlx-lm` library.

        """
        from mlx_lm import generate

        return generate(
            self.model,
            self.mlx_tokenizer,
            self.type_adapter.format_input(model_input),
            logits_processors=self.type_adapter.format_output_type(output_type),
            **kwargs,
        )

    def generate_stream(self, model_input, output_type=None, **kwargs):
        """Stream text using `mlx-lm`.

        Arguments
        ---------
        model_input
            The prompt passed by the user.
        output_type
            The logits processor provided.
        kwargs
            The inference kwargs that can be passed to the `mlx-lm` library.

        """
        from mlx_lm import stream_generate

        for token in stream_generate(
            self.model,
            self.mlx_tokenizer,
            self.type_adapter.format_input(model_input),
            logits_processors=self.type_adapter.format_output_type(output_type),
            **kwargs,
        ):
            yield token


def from_mlxlm(model: "nn.Module", tokenizer: "PreTrainedTokenizer") -> MLXLM:
    return MLXLM(model, tokenizer)
