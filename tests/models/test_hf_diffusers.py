import numpy as np
from PIL.Image import Image as PILImage

from outlines.models.hf_diffusers import HuggingFaceDiffuser

MODEL = "hf-internal-testing/tiny-stable-diffusion-torch"


def test_stable_diffusion():
    model = HuggingFaceDiffuser(MODEL)

    image = model("test")
    assert isinstance(image, PILImage)

    images = model("test", samples=3)
    assert isinstance(images, np.ndarray)
    assert len(images) == 3
    for img in images:
        assert isinstance(image, PILImage)
