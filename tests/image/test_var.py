import pytest
from PIL.Image import Image as PILImage

import outlines
from outlines.image.var import ImageConstant
from outlines.text.var import Variable


def test_cast():
    with pytest.raises(TypeError):
        outlines.as_image("")

    with pytest.raises(TypeError):
        outlines.as_image(Variable())

    with pytest.raises(TypeError):
        outlines.as_image(ImageConstant(""))

    img = PILImage()
    s = outlines.as_image(img)
    assert isinstance(s, ImageConstant)
    assert isinstance(s.value, type(img))

    i = ImageConstant(img)
    outlines.as_image(i)

    i = outlines.image.image()
    outlines.as_image(i)
