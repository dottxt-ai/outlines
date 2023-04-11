from outlines.text.prompt import prompt


def generation(name: str):
    """Decorator that allows to simplify calls to image generation models."""
    provider_name = name.split("/")[0]
    model_name = name[len(provider_name) + 1 :]

    if provider_name == "hf":
        from outlines.image.models.hugging_face import HFDiffuser

        generative_model = HFDiffuser(model_name)  # type:ignore
    else:
        raise NameError(f"The model provider {provider_name} is not available.")

    def decorator(fn):
        prompt_fn = prompt(fn)

        def wrapper(*args, **kwargs):
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
