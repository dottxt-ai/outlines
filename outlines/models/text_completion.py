"""Router for text completion models."""
from .hf_transformers import HuggingFaceCompletion
from .openai import OpenAIChatCompletion, OpenAITextCompletion

hf = HuggingFaceCompletion


def openai(model_name: str, *args, **kwargs):
    """Dispatch the OpenAI model names to their respective completion API.

    This ensures that chat completion models can also be called as text
    completion models (with no instruction and no history).

    Parameters
    ----------
    model_name
        The name of the model in OpenAI's API.

    """
    if "text-" in model_name:
        return OpenAITextCompletion(model_name, *args, **kwargs)
    elif "gpt-" in model_name:
        return OpenAIChatCompletion(model_name, *args, **kwargs)
    else:
        raise NameError(
            f"The model {model_name} requested is not available. Only the completion and chat completion models are available for OpenAI."
        )
