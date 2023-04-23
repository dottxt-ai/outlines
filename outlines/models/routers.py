"""Route model names to their corresponding implementation."""
import functools
from typing import Callable, Dict, Tuple

import outlines.models as models


def image_generation(model_path: str) -> Callable:
    """Return the model and model name corresponding to the model path.

    Parameters
    ----------
    model_path
        A string of the form "model_provider/model_name"

    Returns
    -------
    The model builder with bound model name.

    """

    registry: Dict[str, Callable] = {
        "hf": models.HuggingFaceDiffuser,
        "openai": models.OpenAIImageGeneration,
    }

    provider, model_name = parse_model_path(model_path)

    try:
        model = registry[provider]
    except KeyError:
        raise ValueError(f"The model provider {provider} is not available.")

    return functools.partial(model, model_name)


def parse_model_path(model_path: str) -> Tuple[str, str]:
    """Parse a model path in the form 'provider/model_name'"""

    if "/" not in model_path:
        raise ValueError("Model names must be in the form 'provider_name/model_name'")

    provider_name = model_path.split("/")[0]
    model_name = model_path[len(provider_name) + 1 :]

    return provider_name, model_name
