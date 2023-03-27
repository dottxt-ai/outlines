from outlines.image.models.model import ImageModel

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    raise ImportError(
        "You need to install `torch` and `diffusers` to run the StableDiffusion model."
    )


class StableDiffusion(ImageModel):
    """A `StableDiffusion` distributed random image."""

    def __init__(self, name=None):
        super().__init__(name)

    def sample(self, prompt):
        """Use HuggingFace's `StableDiffusion` pipeline to sample a new image."""
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to("cuda")
        image = pipe(prompt).images[0]

        return image
