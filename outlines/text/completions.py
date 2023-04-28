from typing import Any, Callable, Dict, List, Optional, Tuple

import outlines.models as models
import outlines.text as text


def get_model_from_path(
    model_path: str,
    *,
    stop_at: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Callable:
    """Obtain a text completion provider model object from a model path.

    Parameters
    ----------
    model_path
        A string of the form "model_provider/model_name"
    stop_at
        A list of tokens which, when found, stop the generation.
    max_tokens
        The maximum number of tokens to generate.
    temperature
        Value used to module the next token probabilities.

    """
    if "/" not in model_path:
        raise ValueError("Model names must be in the form 'provider_name/model_name'")

    provider_name = model_path.split("/")[0]
    model_name = model_path[len(provider_name) + 1 :]

    try:
        model_cls = getattr(models.text_completion, provider_name)
    except KeyError:
        raise ValueError(f"The model provider {provider_name} is not available.")

    llm = model_cls(
        model_name, stop_at=stop_at, max_tokens=max_tokens, temperature=temperature
    )

    return llm


def completion(
    model_path: str,
    *,
    stop_at: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Callable:
    """Decorator that simplifies calls to language models.

    Prompts that are passed to language models are often rendered templates,
    and the workflow typically looks like:

    >>> import outlines
    >>> from outlines.models import OpenAICompletion
    >>>
    >>> llm = OpenAICompletion("davinci")
    >>> tpl = "I have a {{question}}"
    >>> prompt = outlines.render(tpl, question="How are you?")
    >>> answer = llm(prompt)

    While explicit, these 4 lines have the following defaults:

    1. The prompt is hidden;
    2. The language model instantiation is far from the prompt; prompt templates
    are however attached to a specific language model call.
    3. The intent behind the language model call is hidden.

    To encapsulate the logic behind language model calls, we thus define the
    template prompt inside a function and decorate the function with a model
    specification. When that function is called, the template is rendered using
    the arguments passed to the function, and the rendered prompt is passed to
    a language model instantiated with the arguments passed to the decorator.

    The previous example is equivalent to the following:

    >>> import outlines.text as text
    >>>
    >>> @text.completion("openai/davinci")
    ... def answer(question):
    ...     "I have a {{question}}"
    ...
    >>> answer, _ = answer("How are you?")

    Decorated functions return two objects: the first represents the output of
    the language model call, the second represents the concatenation of the
    rendered prompt with the output of the language model call. The latter can
    be used in context where one expands an initial prompt with recursive calls
    to language models.

    Parameters
    ----------
    model_path
        A string of the form "model_provider/model_name"
    stop_at
        A list of tokens which, when found, stop the generation.
    max_tokens
        The maximum number of tokens to generate.
    temperature
        Value used to module the next token probabilities.

    """
    llm = get_model_from_path(
        model_path, stop_at=stop_at, max_tokens=max_tokens, temperature=temperature
    )

    def decorator(fn: Callable):
        prompt_fn = text.prompt(fn)

        def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Tuple[str, str]:
            """Call the generative model with the rendered template.

            Building prompts with recursive calls to language models is common
            in prompt engineering, we thus return both the raw answer from the
            language model as well as the rendered prompt including the answer.

            Returns
            -------
            A tuple that contains the result of the language model call, and the
            rendered prompt concatenated with the result of the language model
            call.

            """
            prompt = prompt_fn(*args, **kwargs)
            result = llm(prompt)
            return result, prompt + result

        return wrapper

    return decorator
