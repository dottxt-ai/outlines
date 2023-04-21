import functools
import inspect
import re
import textwrap
from typing import Callable, Optional


class Tool:
    """An Outlines tool definition.

    Wraps python functions and automatically extracts their name, definition and
    description.

    Attributes
    ----------
    fn
        The function being wrapped.
    name
        The name of the function.
    description
        The function's description as read in the first line of the docstring,
        or passed at instantiation.
    signature
        The function's signature in a string format.

    """

    def __init__(self, fn: Callable, description: Optional[str] = None):
        """

        Parameters
        ----------
        fn
            The function being wrapped.
        description
            The description contained in the function's docstring will
            be overriden by this value.

        """
        if not callable(fn):
            raise TypeError("`Tool` must be instantiated by passing a callable.")
        self.fn = fn

        # Get the function's name
        if not hasattr(fn, "__name__"):
            self.name = type(fn).__name__
        else:
            self.name = fn.__name__

        # When unspecified, the docstring's first line is used as a description
        if description is None:
            docstring = inspect.getdoc(fn)
            if docstring is None:
                description = None
            else:
                description = docstring.split("\n")[0].strip()

        self.description = description

        # Get the function's source code, without the decorator if present.
        source = textwrap.dedent(inspect.getsource(fn))
        re_search = re.search(re.compile(r"(\bdef\b.*)", re.DOTALL), source)
        if re_search is not None:
            source = re_search.group(0)
        else:
            raise TypeError("Could not read the function's source code")
        self.source = source

        # Extract the signature part of the function's source code
        re_search = re.search(re.compile(r"\(([^)]+)\)"), source)
        if re_search is None:
            self.signature = ""
        else:
            self.signature = re_search.group(1)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def tool(fn=None, *, description=None):
    """Decorator to designate a function as a tool.

    Parameters
    ----------
    description
        The description contained in the function's docstring will
        be overriden by this value.

    Returns
    -------
    A `Tool` object which will call the decorated function when called.

    Examples
    --------

    Define a simple function, its description will be read from the docstring's
    first line:

    >>> @outlines.tool
    ... def repeat(word: str, n: int):
    ...     "Repeat the word n times"
    ...     return words * n

    We can also override the description:

    >>> @outlines.tool(description="n time the word")
    ... def repeat(word: str, n: int):
    ...     "Repeat the word n times"
    ...     return words * n

    """

    if fn is not None:
        return Tool(fn, description=description)
    else:
        return functools.partial(tool, description=description)
