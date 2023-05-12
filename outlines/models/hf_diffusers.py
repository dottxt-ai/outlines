"""Integration with HuggingFace's `diffusers` library."""
from PIL.Image import Image as PILImage

from outlines.caching import cache


def HuggingFaceDiffuser(model_name: str) -> PILImage:
    """Create a function that will call a stable diffusion pipeline.

    Parameters
    ----------
    model_name: str
        The name of the model as listed on HuggingFace's models page.

    """

    def call(prompt: str, samples: int = 1) -> str:
        return call_stable_diffusion_pipeline(model_name, prompt, samples)

    return call


@cache
def call_stable_diffusion_pipeline(
    model_name: str, prompt: str, samples: int
) -> PILImage:
    """Build and call the Stable Diffusion pipeline."""
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    image = pipe(prompt, num_images_per_prompt=samples).images[0]

    return image
