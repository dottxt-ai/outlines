from typing import Any, Callable, Dict, List

from PIL.Image import Image as PILImage

import outlines.models.routers as routers
from outlines.text import prompt


def generation(model_path: str) -> Callable:
    """Decorator that allows to simplify calls to image generation models."""
    generative_model_builder = routers.image_generation(model_path)
    generative_model = generative_model_builder()

    def decorator(fn: Callable):
        prompt_fn = prompt(fn)

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
