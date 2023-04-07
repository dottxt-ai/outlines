import collections
import inspect
from typing import Dict, Union

from mako.runtime import Context
from mako.template import Template

from outlines.text.var import StringVariable


class OutlinesEncodingBuffer:
    """An encoding buffer for Mako's templating engine.

    This is a modified version of Mako's `FastEncodingBuffer`. It build outlines
    graph when the template is rendered with `StringVariable`s.

    """

    def __init__(self, encoding=None, errors="strict"):
        self.data = collections.deque()
        self.encoding = encoding
        self.delim = ""
        self.errors = errors
        self.write = self.data.append

    def truncate(self):
        self.data = collections.deque()
        self.write = self.data.append

    def get_value(self):
        if self.encoding:
            return self.delim.join(self.data).encode(self.encoding, self.errors)
        else:
            output = ""
            for d in self.data:
                if isinstance(d, StringVariable):
                    output = output + d
                else:
                    output = output + str(d)
            return output


def render(
    template: str, **values: Dict[str, Union[str, StringVariable]]
) -> Union[str, StringVariable]:
    r"""Parse a Mako template and translate it into an Outlines graph.

    Examples
    --------

    Outlines follow Mako's syntax

    >>> import outlines
    >>> outline = outlines.render("I like ${food} and ${sport}", food="tomatoes", sport="tennis")
    I like tomatoes and tennis

    When a variable in the template is assigne a `StringVariable` value, the
    `render` function builds the corresponding outlines graph and returns a
    `StringVariable`:

    >>> s = outlines.text.string()
    >>> outlines.render("I like ${food}", food=food)
    <StringVariable>

    It is also possible to use control flow inside templates:

    >>> examples = ["one", "two", "three"]
    >>> outlines = outlines.render(
    ...     '''
    ...     % for example in examples:
    ...     Example: ${example}
    ...     % endfor
    ...     ''',
    ...     examples=examples
    ... )

    Parameters
    ----------
    template
        A string that contains a template written in the Mako syntax.
    **values
        Map from the variables in the template to their value.

    Returns
    -------
    A string when the values are all strings, a `StringVariable` otherwise.

    """
    buf = OutlinesEncodingBuffer()
    ctx = Context(buf, **values)

    outline = inspect.cleandoc(template)
    mako_template = Template(outline, default_filters=[])
    mako_template.render_context(ctx)

    return buf.get_value()
