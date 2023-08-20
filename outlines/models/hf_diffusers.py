"""Integration with Hugging Face's `diffusers` library."""
import functools
from typing import List, Union

import numpy as np
from PIL.Image import Image as PILImage

import outlines


def HuggingFaceDiffuser(model_name: str) -> PILImage:
    """Create a function that will call a stable diffusion pipeline.

    Parameters
    ----------
    model_name: str
        The name of the model as listed on Hugging Face's models page.

    """

    def call(prompt: Union[str, List[str]], samples: int = 1) -> str:
        if isinstance(prompt, str):
            prompt = [prompt]

        results = call_stable_diffusion_pipeline(model_name, prompt, samples)

        return results

    return call


@functools.partial(outlines.vectorize, signature="(),(m),()->(m,s)")
def call_stable_diffusion_pipeline(
    model_name: str, prompt: List[str], samples: int
) -> PILImage:
    """Build and call the Stable Diffusion pipeline.

    We convert the returned image
    """
    import torch
    from diffusers import StableDiffusionPipeline

    # Pipelines don't accept NumPy arrays
    prompt = list(prompt)

    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    elif torch.backends.mps.is_available():
        pipe = pipe.to("mps")

    images = pipe(prompt, num_images_per_prompt=samples).images
    if not isinstance(images, list):
        images = [images]

    array = np.empty((samples,), dtype="object")
    for idx, image in enumerate(images):
        array[idx] = image

    return np.atleast_2d(array)
