"""Outlines is a Probabilistic Generative Model Programming language.

Outlines allows to build and evaluate graphs that represent interactions between
user-defined strings (usually called "prompts"), images, and generative models.

Most generative models are probabilistic, their outputs are random variables
whose distribution is determined by the generative model. Chained generative
models are thus probabilistic programs [1]_. By default, compiling an Outlines
graph returns a callable object which will yield different values each time it
is called.

Deterministic decoding methods are implemented as graph transformations: they
take a probabilistic program as an input and return a graph that represents the
decoding process. Compiling these graphs will produce a callable that returns
the same value each time it is called.

Outlines supports plugins as long as they admit text/images as an input and
return strings and/or images. They are represented by operators in the graph.

The design of Outlines was heavily inspired by `Aesara <https://github.com/aesara-devs/aesara>`_,
a library for defining, optimizing and evaluating mathematical expressions
involving multi-dimensional arrays. A complete integration would be desirable
and is not excluded.


References
----------
.. [1] Dohan, David, et al. "Language model cascades." arXiv preprint arXiv:2207.10342 (2022).

"""
from outlines.compile import compile
from outlines.text import script, string

__all__ = [
    "compile",
    "script",
    "string",
]
