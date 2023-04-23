import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from jinja2 import StrictUndefined, Template

import outlines.models.routers as routers


def render(template: str, **values: Optional[Dict[str, Any]]) -> str:
    r"""Parse a Jinaj2 template and translate it into an Outlines graph.

    This function removes extra whitespaces and linebreaks from templates to
    allow users to enter prompt more naturally than if they used Python's
    constructs directly. See the examples for a detailed explanation.

    Examples
    --------

    Outlines follow Jinja2's syntax

    >>> import outlines
    >>> outline = outlines.render("I like {{food}} and {{sport}}", food="tomatoes", sport="tennis")
    I like tomatoes and tennis

    If the first line of the template is empty, `render` removes it

    >>> from outlines import render
    >>>
    >>> tpl = '''
    ... A new string'''
    >>> tpl
    ... '\nA new string'
    >>> render(tpl)
    ... 'a new string'

    Similarly, `render` ignores linebreaks introduced by placing the closing quotes
    underneath the text:

    >>> tpl = '''
    ... A new string
    ... '''
    >>> tpl
    ... '\nA new string\n'
    >>> render(tpl)
    ... 'A new string'

    `render` removes the identation in docstrings. This is particularly important
    when using prompt functions

    >>> tpl = '''
    ...    a string
    ...    and another string'''
    >>> tpl
    ... '\n   a string\n   and another string'
    >>> render(tpl)
    ... 'a string\nand another string'

    The indentation of the first line is assumed to be the same as the second line's

    >>> tpl = '''a string
    ...     and another'''
    >>> tpl
    ... 'a string\n    and another'
    >>> render(tpl)
    ... 'a string\nand another'

    To get a different indentation for the first and the second line, we can start the
    prompt on the string's second line:

    >>> tpl = '''
    ... First line
    ...   Second line'''
    >>> render(tpl)
    ... 'First Line\n  Second Line'

    Parameters
    ----------
    template
        A string that contains a template written with the Jinja2 syntax.
    **values
        Map from the variables in the template to their value.

    Returns
    -------
    A string that contains the rendered template.

    """

    # Dedent, and remove extra linebreak
    template = inspect.cleandoc(template)

    # Remove extra whitespaces, except those that immediately follow a newline symbol.
    # This is necessary to avoid introducing whitespaces after backslash `\` characters
    # used to continue to the next line without linebreak.
    template = re.sub(r"(?![\r\n])(\b\s+)", " ", template)

    mako_template = Template(
        template,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        undefined=StrictUndefined,
    )
    return mako_template.render(**values)


def prompt(fn: Callable) -> Callable:
    """Decorate a function that contains a prompt template.

    This allows to define prompts in the docstring of a function and ease their
    manipulation by providing some degree of encapsulation. It uses the `render`
    function internally to render templates.

    >>> import outlines
    >>>
    >>> @outlines.prompt
    >>> def build_prompt(question):
    ...    "I have a ${question}"
    ...
    >>> prompt = build_prompt("How are you?")

    """

    sig = inspect.signature(fn)

    # The docstring contains the template that will be rendered to be used
    # as a prompt to the language model.
    docstring = fn.__doc__
    if docstring is None:
        raise TypeError("Could not find a template in the function's docstring.")

    def wrapper(*args: Optional[List[str]], **kwargs: Optional[Dict[str, str]]) -> str:
        """Render and return the template.

        Returns
        -------
        The rendered template as a Python ``str``.

        """
        template = cast(str, docstring)  # for typechecking
        bound_arguments = sig.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return render(template, **bound_arguments.arguments)

    return wrapper


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
    >>> @outlines.completion("openai/davinci")
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
    llm_builder = routers.text_completion(model_path)
    llm = llm_builder(stop_at=stop_at, max_tokens=max_tokens, temperature=temperature)

    def decorator(fn: Callable):
        prompt_fn = prompt(fn)

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
