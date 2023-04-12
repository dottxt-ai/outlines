try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    raise ImportError(
        "You need to install `torch` and `diffusers` to run the StableDiffusion model."
    )


class HFDiffuser:
    """A `StableDiffusion` distributed random image."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, prompt: str) -> str:
        """Use HuggingFace's `StableDiffusion` pipeline to sample a new image."""
        pipe = StableDiffusionPipeline.from_pretrained(self.model_name)
        pipe = pipe.to("cuda")
        image = pipe(prompt).images[0]

        return image
