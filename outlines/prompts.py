import functools
import inspect
import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, cast

import jinja2
import pydantic


@dataclass
class Prompt:
    """Represents a prompt function.

    We return a `Prompt` class instead of a simple function so the
    template defined in prompt functions can be accessed.

    """

    template: jinja2.Template
    signature: Optional[inspect.Signature]

    def __call__(self, *args, **kwargs) -> str:
        """Render and return the template.

        Returns
        -------
        The rendered template as a Python ``str``.

        """
        if self.signature is not None:
            bound_arguments = self.signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            return self.template.render(**bound_arguments.arguments)
        else:
            return self.template.render(**kwargs)

    @classmethod
    def from_str(cls, content: str):
        """
        Create an instance of the class from a string.

        Parameters
        ----------
        content : str
            The string content to be converted into a template.

        Returns
        -------
        An instance of the class with the provided content as a template.
        """
        return cls(cls._template_from_str(content), None)

    @classmethod
    def from_file(cls, path: Path):
        """
        Create a Prompt instance from a file containing a Jinja template.

        Note: This method does not allow to include and inheritance to reference files
        that are outside the folder or subfolders of the file given to `from_file`.

        Parameters
        ----------
        path : Path
            The path to the file containing the Jinja template.

        Returns
        -------
        Prompt
            An instance of the Prompt class with the template loaded from the file.
        """
        # We don't use a `Signature` here because it seems not feasible to infer one from a Jinja2 environment that is
        # split across multiple files (since e.g. we support features like Jinja2 includes and template inheritance)
        return cls(cls._template_from_file(path), None)

    @classmethod
    def _template_from_str(_, content: str) -> jinja2.Template:
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

        env = jinja2.Environment(
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

        return env.from_string(cleaned_template)

    @classmethod
    def _template_from_file(_, path: Path) -> jinja2.Template:
        file_directory = os.path.dirname(os.path.abspath(path))
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(file_directory),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            undefined=jinja2.StrictUndefined,
        )
        return env.get_template(os.path.basename(path))


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

    template = Prompt._template_from_str(cast(str, docstring))

    return Prompt(template, signature)


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


@get_schema.register(type(pydantic.BaseModel))
def get_schema_pydantic(model: Type[pydantic.BaseModel]):
    """Return the schema of a Pydantic model."""
    if not isinstance(model, type(pydantic.BaseModel)):
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
