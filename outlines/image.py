from typing import Any, Callable, Dict, List

from PIL.Image import Image as PILImage

import outlines.models as models
import outlines.text as text


def generation(model_path: str) -> Callable:
    """Decorator that allows to simplify calls to image generation models."""

    if "/" not in model_path:
        raise ValueError("Model names must be in the form 'provider_name/model_name'")

    provider_name = model_path.split("/")[0]
    model_name = model_path[len(provider_name) + 1 :]

    try:
        model_cls = getattr(models.image_generation, provider_name)
    except KeyError:
        raise ValueError(f"The model provider {provider_name} is not available.")

    generative_model = model_cls(model_name)

    def decorator(fn: Callable):
        prompt_fn = text.prompt(fn)

        def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> PILImage:
            """Call the Diffuser with the rendered template.

            Returns
            -------
            A `PIL.Image` instance that represents the generated image.

            """
            prompt = prompt_fn(*args, **kwargs)
            result = generative_model(prompt)
            return result

        return wrapper

    return decorator
