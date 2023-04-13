import functools

from PIL.Image import Image as PILImage


def HuggingFaceDiffuser(model_name: str) -> PILImage:
    """Create a function that will call a stable diffusion pipeline.

    Parameters
    ----------
    model_name: str
        The name of the model as listed on HuggingFace's models page.

    """

    @functools.lru_cache
    def call(prompt: str) -> str:
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(model_name)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        image = pipe(prompt).images[0]

        return image
