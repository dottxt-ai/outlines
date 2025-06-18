"""Create templates to easily build prompts."""

import base64
import functools
import inspect
import json
import os
import re
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, cast
import warnings

import jinja2
from PIL import Image
from pydantic import BaseModel


@dataclass
class Vision:
    """Contains the input for a vision model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports vision.

    Parameters
    ----------
    prompt
        The prompt to use to generate the response.
    image
        The image to use to generate the response.

    """
    prompt: str
    image: Image.Image

    def __post_init__(self):
        image = self.image

        if not image.format:
            raise TypeError(
                "Could not read the format of the image passed to the model."
            )

        buffer = BytesIO()
        image.save(buffer, format=image.format)
        self.image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        self.image_format = f"image/{image.format.lower()}"


@dataclass
class Template:
    """Represents a prompt template.

    We return a `Template` class instead of a simple function so the
    template can be accessed by callers.

    """
    template: jinja2.Template
    signature: Optional[inspect.Signature]

    def __call__(self, *args, **kwargs) -> str:
        """Render and return the template.

        Returns
        -------
        str
            The rendered template as a Python string.

        """
        if self.signature is not None:
            bound_arguments = self.signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            return self.template.render(**bound_arguments.arguments)
        else:
            return self.template.render(**kwargs)

    @classmethod
    def from_string(cls, content: str, filters: Dict[str, Callable] = {}):
        """Create a `Template` instance from a string containing a Jinja
        template.

        Parameters
        ----------
        content : str
            The string content to be converted into a template.

        Returns
        -------
        Template
            An instance of the class with the provided content as a template.

        """
        return cls(build_template_from_string(content, filters), None)

    @classmethod
    def from_file(cls, path: Path, filters: Dict[str, Callable] = {}):
        """Create a `Template` instance from a file containing a Jinja
        template.

        Note: This method does not allow to include and inheritance to
        reference files that are outside the folder or subfolders of the file
        given to `from_file`.

        Parameters
        ----------
        path : Path
            The path to the file containing the Jinja template.

        Returns
        -------
        Template
            An instance of the Template class with the template loaded from the
            file.

        """
        # We don't use a `Signature` here because it seems not feasible to
        # infer one from a Jinja2 environment that is
        # split across multiple files (since e.g. we support features like
        # Jinja2 includes and template inheritance)
        return cls(build_template_from_file(path, filters), None)


def build_template_from_string(
    content: str, filters: Dict[str, Callable] = {}
) -> jinja2.Template:
    # Dedent, and remove extra linebreak
    cleaned_template = inspect.cleandoc(content)

    # Add linebreak if there were any extra linebreaks that
    # `cleandoc` would have removed
    ends_with_linebreak = content.replace(" ", "").endswith("\n\n")
    if ends_with_linebreak:
        cleaned_template += "\n"

    # Remove extra whitespaces, except those that immediately follow a newline symbol.
    # This is necessary to avoid introducing whitespaces after backslash `\` characters
    # used to continue to the next line without linebreak.
    cleaned_template = re.sub(r"(?![\r\n])(\b\s+)", " ", cleaned_template)

    env = create_jinja_env(None, filters)

    return env.from_string(cleaned_template)


def build_template_from_file(
    path: Path, filters: Dict[str, Callable] = {}
) -> jinja2.Template:
    file_directory = os.path.dirname(os.path.abspath(path))
    env = create_jinja_env(jinja2.FileSystemLoader(file_directory), filters)

    return env.get_template(os.path.basename(path))


def prompt(
    fn: Optional[Callable] = None,
    filters: Dict[str, Callable] = {},
) -> Callable:
    """Decorate a function that contains a prompt template.

    This allows to define prompts in the docstring of a function and simplify their
    manipulation by providing some degree of encapsulation. It uses the `render`
    function internally to render templates.

    ```pycon
    >>> import outlines
    >>>
    >>> @outlines.prompt
    >>> def build_prompt(question):
    ...    "I have a ${question}"
    ...
    >>> prompt = build_prompt("How are you?")
    ```

    This API can also be helpful in an "agent" context where parts of the prompt
    are set when the agent is initialized and never modified later. In this situation
    we can partially apply the prompt function at initialization.

    ```pycon
    >>> import outlines
    >>> import functools as ft
    ...
    >>> @outlines.prompt
    ... def solve_task(name: str, objective: str, task: str):
    ...     \"""Your name is {{name}}.
    ...     Your overall objective is to {{objective}}.
    ...     Please solve the following task: {{task}}
    ...     \"""
    ...
    >>> hal = ft.partial(solve_task, "HAL", "Travel to Jupiter")
    ```

    Additional Jinja2 filters can be provided as keyword arguments to the decorator.

    ```pycon
    >>> def reverse(s: str) -> str:
    ...     return s[::-1]
    ...
    >>> @outlines.prompt(filters={ 'reverse': reverse })
    ... def reverse_prompt(text):
    ...     \"""{{ text | reverse }}\"""
    ...
    >>> prompt = reverse_prompt("Hello")
    >>> print(prompt)
    ... "olleH"
    ```

    Returns
    -------
    A `Template` callable class which will render the template when called.

    """
    warnings.warn(
        "The @prompt decorator is deprecated and will be removed in outlines 1.1.0. "
        "Instead of using docstring templates, please use Template.from_file() to "
        "load your prompts from separate template files, or a simple Python function "
        "that returns text. This helps keep prompt content separate from code and is "
        "more maintainable.",
        DeprecationWarning,
        stacklevel=2,
    )

    if fn is None:
        return lambda fn: prompt(fn, cast(Dict[str, Callable], filters))

    signature = inspect.signature(fn)

    # The docstring contains the template that will be rendered to be used
    # as a prompt to the language model.
    docstring = fn.__doc__
    if docstring is None:
        raise TypeError("Could not find a template in the function's docstring.")

    template = build_template_from_string(cast(str, docstring), filters)

    return Template(template, signature)


def create_jinja_env(
    loader: Optional[jinja2.BaseLoader], filters: Dict[str, Callable]
) -> jinja2.Environment:
    """Create a new Jinja environment.

    The Jinja environment is loaded with a set of pre-defined filters:
    - `name`: get the name of a function
    - `description`: get a function's docstring
    - `source`: get a function's source code
    - `signature`: get a function's signature
    - `args`: get a function's arguments
    - `schema`: display a JSON Schema

    Users may pass additional filters, and/or override existing ones.

    Parameters
    ----------
    loader
       An optional `BaseLoader` instance
    filters
       A dictionary of filters, map between the filter's name and the
       corresponding function.

    """
    env = jinja2.Environment(
        loader=loader,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )

    env.filters["name"] = get_fn_name
    env.filters["description"] = get_fn_description
    env.filters["source"] = get_fn_source
    env.filters["signature"] = get_fn_signature
    env.filters["schema"] = get_schema
    env.filters["args"] = get_fn_args

    # The filters passed by the user may override the
    # pre-defined filters.
    for name, filter_fn in filters.items():
        env.filters[name] = filter_fn

    return env


def get_fn_name(fn: Callable):
    """Returns the name of a callable."""
    if not callable(fn):
        raise TypeError("The `name` filter only applies to callables.")

    if not hasattr(fn, "__name__"):
        name = type(fn).__name__
    else:
        name = fn.__name__

    return name


def get_fn_args(fn: Callable):
    """Returns the arguments of a function with annotations and default values if provided."""
    if not callable(fn):
        raise TypeError("The `args` filter only applies to callables.")

    arg_str_list = []
    signature = inspect.signature(fn)
    arg_str_list = [str(param) for param in signature.parameters.values()]
    arg_str = ", ".join(arg_str_list)
    return arg_str


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
    else:  # pragma: no cover
        raise TypeError("Could not read the function's source code")

    return source


def get_fn_signature(fn: Callable):
    """Return the signature of a callable."""
    if not callable(fn):
        raise TypeError("The `source` filter only applies to callables.")

    source = textwrap.dedent(inspect.getsource(fn))
    re_search = re.search(re.compile(r"\(([^)]+)\)"), source)
    if re_search is None:  # pragma: no cover
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
