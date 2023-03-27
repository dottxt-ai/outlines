from functools import singledispatch

from PIL.Image import Image as PILImage

from outlines.graph import Variable


class ImageVariable(Variable):
    """Subclass to add the image operators to `Variable`."""

    def __init__(self, owner=None, index=None, name=None):
        super().__init__(owner, index, name)


image = ImageVariable


class ImageConstant(ImageVariable):
    """Constant `ImageVariable` that corresponds to user input."""

    def __init__(self, value, name=None):
        if not isinstance(value, PILImage):
            raise TypeError(
                "`ImageConstant` values must be instances of `pillow.Image`."
            )

        self.value = value
        super().__init__(name=name)

    def __str__(self):
        if self.name is not None:
            name = self.name
        else:
            name = "ImageConstant"
        return f"{name}{{'{self.value}'}}"


@singledispatch
def as_image(x, name=None):
    """Convert `x` into an equivalent `StringVariable`

    This function can be used to turn `pillow.Image` instances into an
    `ImageVariable`.

    Parameters
    ----------
    x
        The object that will we converted into a `ImageVariable`.
    name
        If a new `ImageVariable` instance is created it will be attributed this
        name.

    """
    raise TypeError(f"{x} cannot be cast into a string")


@as_image.register(PILImage)
def as_image_Image(x, name=None):
    return ImageConstant(x, name)


@as_image.register(ImageVariable)
def as_image_ImageConstant(x, name=None):
    return x
