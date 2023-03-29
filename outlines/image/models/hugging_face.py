from outlines.image.models.model import ImageModel

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    raise ImportError(
        "You need to install `torch` and `diffusers` to run the StableDiffusion model."
    )


class HFDiffuser(ImageModel):
    """A `StableDiffusion` distributed random image."""

    def __init__(self, model_name: str, name=None):
        self.model_name = model_name
        super().__init__(name)

    def sample(self, prompt):
        """Use HuggingFace's `StableDiffusion` pipeline to sample a new image."""
        pipe = StableDiffusionPipeline.from_pretrained(self.model_name)
        pipe = pipe.to("cuda")
        image = pipe(prompt).images[0]

        return image
