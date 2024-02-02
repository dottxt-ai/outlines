import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel


@dataclass
class Prompt:
    """Represents a prompt function.

    We return a `Prompt` class instead of a simple function so the
    template defined in prompt functions can be accessed.

    """

    template: str
    signature: inspect.Signature

    def __post_init__(self):
        self.parameters: List[str] = list(self.signature.parameters.keys())

    def __call__(self, *args, **kwargs) -> str:
        """Render and return the template.

        Returns
        -------
        The rendered template as a Python ``str``.

        """
        bound_arguments = self.signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return render(self.template, **bound_arguments.arguments)

    def __str__(self):
        return self.template


def prompt(fn: Callable) -> Prompt:
    """Decorate a function that contains a prompt template.

    This allows to define prompts in the docstring of a function and simplify their
    manipulation by providing some degree of encapsulation. It uses the `render`
    function internally to render templates.

    >>> import outlines
    >>>
    >>> @outlines.prompt
    >>> def build_prompt(question):
    ...    "I have a ${question}"
    ...
    >>> prompt = build_prompt("How are you?")

    This API can also be helpful in an "agent" context where parts of the prompt
    are set when the agent is initialized and never modified later. In this situation
    we can partially apply the prompt function at initialization.

    >>> import outlines
    >>> import functools as ft
    ...
    >>> @outlines.prompt
    ... def solve_task(name: str, objective: str, task: str):
    ...     '''Your name is {{name}}.
    ..      Your overall objective is to {{objective}}.
    ...     Please solve the following task: {{task}}
    ...     '''
    ...
    >>> hal = ft.partial(solve_task, "HAL", "Travel to Jupiter")

    Returns
    -------
    A `Prompt` callable class which will render the template when called.

    """

    signature = inspect.signature(fn)

    # The docstring contains the template that will be rendered to be used
    # as a prompt to the language model.
    docstring = fn.__doc__
    if docstring is None:
        raise TypeError("Could not find a template in the function's docstring.")

    template = cast(str, docstring)

    return Prompt(template, signature)


def render(template: str, **values: Optional[Dict[str, Any]]) -> str:
    r"""Parse a Jinaj2 template and translate it into an Outlines graph.

    This function removes extra whitespaces and linebreaks from templates to
    allow users to enter prompts more naturally than if they used Python's
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

    If you want to insert a linebreak at the end of the rendered template, you will
    need to leave an empty line at the end of the template:

    >>> tpl = '''
    ... A new string
    ...
    ... '''
    >>> tpl
    ... '\nA new string\n\n'
    >>> render(tpl)
    ... 'A new string\n'

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
    cleaned_template = inspect.cleandoc(template)

    # Add linebreak if there were any extra linebreaks that
    # `cleandoc` would have removed
    ends_with_linebreak = template.replace(" ", "").endswith("\n\n")
    if ends_with_linebreak:
        cleaned_template += "\n"

    # Remove extra whitespaces, except those that immediately follow a newline symbol.
    # This is necessary to avoid introducing whitespaces after backslash `\` characters
    # used to continue to the next line without linebreak.
    cleaned_template = re.sub(r"(?![\r\n])(\b\s+)", " ", cleaned_template)

    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters["name"] = get_fn_name
    env.filters["description"] = get_fn_description
    env.filters["source"] = get_fn_source
    env.filters["signature"] = get_fn_signature
    env.filters["schema"] = get_schema

    jinja_template = env.from_string(cleaned_template)

    return jinja_template.render(**values)


def get_fn_name(fn: Callable):
    """Returns the name of a callable."""
    if not callable(fn):
        raise TypeError("The `name` filter only applies to callables.")

    if not hasattr(fn, "__name__"):
        name = type(fn).__name__
    else:
        name = fn.__name__

    return name


def get_fn_description(fn: Callable):
    """Returns the first line of a callable's docstring."""
    if not callable(fn):
        raise TypeError("The `description` filter only applies to callables.")

    docstring = inspect.getdoc(fn)
    if docstring is None:
        description = ""
    else:
        description = docstring.split("\n")[0].strip()

    return description


def get_fn_source(fn: Callable):
    """Return the source code of a callable."""
    if not callable(fn):
        raise TypeError("The `source` filter only applies to callables.")

    source = textwrap.dedent(inspect.getsource(fn))
    re_search = re.search(re.compile(r"(\bdef\b.*)", re.DOTALL), source)
    if re_search is not None:
        source = re_search.group(0)
    else:
        raise TypeError("Could not read the function's source code")

    return source


def get_fn_signature(fn: Callable):
    """Return the signature of a callable."""
    if not callable(fn):
        raise TypeError("The `source` filter only applies to callables.")

    source = textwrap.dedent(inspect.getsource(fn))
    re_search = re.search(re.compile(r"\(([^)]+)\)"), source)
    if re_search is None:
        signature = ""
    else:
        signature = re_search.group(1)

    return signature


@functools.singledispatch
def get_schema(model: Any):
    raise NotImplementedError(
        f"No schema rendering function defined for type {type(model)}."
    )


@get_schema.register(dict)
def get_schema_dict(model: Dict):
    """Return a pretty-printed dictionary"""
    return json.dumps(model, indent=2)


@get_schema.register(type(BaseModel))
def get_schema_pydantic(model: Type[BaseModel]):
    """Return the schema of a Pydantic model."""
    if not type(model) == type(BaseModel):
        raise TypeError("The `schema` filter only applies to Pydantic models.")

    if hasattr(model, "model_json_schema"):
        def_key = "$defs"
        raw_schema = model.model_json_schema()
    else:  # pragma: no cover
        def_key = "definitions"
        raw_schema = model.schema()

    definitions = raw_schema.get(def_key, None)
    schema = parse_pydantic_schema(raw_schema, definitions)

    return json.dumps(schema, indent=2)


def parse_pydantic_schema(raw_schema, definitions):
    """Parse the output of `Basemodel.[schema|model_json_schema]()`.

    This recursively follows the references to other schemas in case
    of nested models. Other schemas are stored under the "definitions"
    key in the schema of the top-level model.

    """
    simple_schema = {}
    for name, value in raw_schema["properties"].items():
        if "description" in value:
            simple_schema[name] = value["description"]
        elif "$ref" in value:
            refs = value["$ref"].split("/")
            simple_schema[name] = parse_pydantic_schema(
                definitions[refs[2]], definitions
            )
        else:
            simple_schema[name] = f"<{name}>"

    return simple_schema
