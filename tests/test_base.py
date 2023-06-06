import numpy as np
import pytest
from numpy.testing import assert_array_equal

from outlines.base import vectorize


def test_vectorize_docstring():
    def test(x):
        """This is a test docstring"""
        return x

    fn = vectorize(test)
    assert fn.__doc__ == "This is a test docstring"
    assert fn.__name__ == "test"

    async def test(x):
        """This is a test docstring"""
        return x

    fn = vectorize(test)
    assert fn.__doc__ == "This is a test docstring"
    assert fn.__name__ == "test"


def test_vectorize_thunk():
    def thunk():
        """A thunk"""
        return 1

    fn = vectorize(thunk)
    assert fn() == 1

    async def thunk():
        """A thunk"""
        return 1

    fn = vectorize(thunk)
    assert fn() == 1


def test_vectorize_scalar_simple():
    def passthrough(x):
        """A passthrough function."""
        return x

    fn = vectorize(passthrough)

    out_vector = fn(["one", "two", "three"])
    assert_array_equal(out_vector, ["one", "two", "three"])

    out_array = fn([["one", "two"], ["three", "four"]])
    assert_array_equal(out_array, [["one", "two"], ["three", "four"]])

    async def passthrough(x):
        """A passthrough function."""
        return x

    fn = vectorize(passthrough)
    assert fn.__doc__ == "A passthrough function."

    out_vector = fn(["one", "two", "three"])
    assert_array_equal(out_vector, ["one", "two", "three"])

    out_array = fn([["one", "two"], ["three", "four"]])
    assert_array_equal(out_array, [["one", "two"], ["three", "four"]])


def test_vectorize_scalar_multiple_outputs():
    def passthrough_multiple_outputs(x):
        return x, x

    fn = vectorize(passthrough_multiple_outputs)
    out = fn(["one", "two", "three"])
    assert len(out) == 2
    assert_array_equal(out[0], ["one", "two", "three"])
    assert_array_equal(out[1], ["one", "two", "three"])

    async def passthrough_multiple_outputs(x):
        return x, x

    fn = vectorize(passthrough_multiple_outputs)
    out = fn(["one", "two", "three"])
    assert len(out) == 2
    assert_array_equal(out[0], ["one", "two", "three"])
    assert_array_equal(out[1], ["one", "two", "three"])


def test_vectorize_scalar_args():
    def passthrough_args(*args):
        """A passthrough function."""
        result = ""
        for arg in args:
            result += arg
        return result

    fn = vectorize(passthrough_args)

    out_array = fn(["one", "two"], ["1", "2"])
    assert_array_equal(out_array, ["one1", "two2"])

    # Broadcasting
    out_array = fn([["one", "two"], ["three", "four"]], ["1", "2"])
    assert_array_equal(out_array, [["one1", "two2"], ["three1", "four2"]])

    async def passthrough_args(*args):
        """A passthrough function."""
        result = ""
        for arg in args:
            result += arg
        return result

    fn = vectorize(passthrough_args)

    out_array = fn(["one", "two"], ["1", "2"])
    assert_array_equal(out_array, ["one1", "two2"])

    # Broadcasting
    out_array = fn([["one", "two"], ["three", "four"]], ["1", "2"])
    assert_array_equal(out_array, [["one1", "two2"], ["three1", "four2"]])


def test_vectorize_scalar_kwargs():
    def passthrough_kwargs(**kwargs):
        """A passthrough function."""
        result = ""
        for _, value in kwargs.items():
            result += value
        return result

    fn = vectorize(passthrough_kwargs)

    out_array = fn(first=["one", "two"], second=["1", "2"])
    assert_array_equal(out_array, ["one1", "two2"])

    # Broadcasting
    out_array = fn(first=[["one", "two"], ["three", "four"]], second=["1", "2"])
    assert_array_equal(out_array, [["one1", "two2"], ["three1", "four2"]])

    async def passthrough_kwargs(**kwargs):
        """A passthrough function."""
        result = ""
        for _, value in kwargs.items():
            result += value
        return result

    fn = vectorize(passthrough_kwargs)

    out_array = fn(first=["one", "two"], second=["1", "2"])
    assert_array_equal(out_array, ["one1", "two2"])

    # Broadcasting
    out_array = fn(first=[["one", "two"], ["three", "four"]], second=["1", "2"])
    assert_array_equal(out_array, [["one1", "two2"], ["three1", "four2"]])


def test_signature_invalid():
    with pytest.raises(ValueError):
        vectorize(lambda x: x, "(m,)->()")

    with pytest.raises(TypeError, match="wrong number of positional"):
        fn = vectorize(lambda x, y: x, "()->()")
        fn(1, 2)

    with pytest.raises(ValueError, match="wrong number of outputs"):

        def test_multioutput(x, y):
            return x, y

        fn = vectorize(test_multioutput, "(),()->()")
        fn(1, 2)


def test_vectorize_simple():
    def test(x):
        return x

    fn = vectorize(test, "(m)->(m)")
    out = fn([["one", "two", "three"], ["four", "five", "six"]])
    assert_array_equal(out, [["one", "two", "three"], ["four", "five", "six"]])

    fn = vectorize(test, "()->()")
    out = fn([["one", "two", "three"], ["four", "five", "six"]])
    assert_array_equal(out, [["one", "two", "three"], ["four", "five", "six"]])


def test_vectorize_kwargs():
    def passthrough_kwargs(**kwargs):
        """A passthrough function."""
        result = ""
        for _, value in kwargs.items():
            result += value
        return result

    fn = vectorize(passthrough_kwargs, "(),()->()")

    out_array = fn(first=["one", "two"], second=["1", "2"])
    assert_array_equal(out_array, ["one1", "two2"])

    # Broadcasting
    out_array = fn(first=[["one", "two"], ["three", "four"]], second=["1", "2"])
    assert_array_equal(out_array, [["one1", "two2"], ["three1", "four2"]])

    async def passthrough_kwargs(**kwargs):
        """A passthrough function."""
        result = ""
        for _, value in kwargs.items():
            result += value
        return result

    fn = vectorize(passthrough_kwargs, "(),()->()")

    out_array = fn(first=["one", "two"], second=["1", "2"])
    assert_array_equal(out_array, ["one1", "two2"])

    # Broadcasting
    out_array = fn(first=[["one", "two"], ["three", "four"]], second=["1", "2"])
    assert_array_equal(out_array, [["one1", "two2"], ["three1", "four2"]])


def test_vectorize_reduce():
    def test(x):
        return x[0]

    fn = vectorize(test, "(m)->()")
    out = fn([["one", "two", "three"], ["four", "five", "six"]])
    assert_array_equal(out, ["one", "four"])

    fn = vectorize(test, "()->()")
    out = fn([["one", "two", "three"], ["four", "five", "six"]])
    assert_array_equal(out, [["o", "t", "t"], ["f", "f", "s"]])


def test_vectorize_expand():
    def test(x):
        return np.array([x for i in range(3)])

    fn = vectorize(test, "()->(s)")
    out = fn(["one", "two"])
    assert_array_equal(out, [["one", "one", "one"], ["two", "two", "two"]])


def test_vectorize_coroutine_simple():
    async def test(x):
        return x

    fn = vectorize(test, "(m)->(m)")
    out = fn([["one", "two", "three"], ["four", "five", "six"]])
    assert_array_equal(out, [["one", "two", "three"], ["four", "five", "six"]])

    fn = vectorize(test, "()->()")
    out = fn([["one", "two", "three"], ["four", "five", "six"]])
    assert_array_equal(out, [["one", "two", "three"], ["four", "five", "six"]])


def test_vectorize_coroutine_reduce():
    async def test(x):
        return x[0]

    fn = vectorize(test, "(m)->()")
    out = fn([["one", "two", "three"], ["four", "five", "six"]])
    assert_array_equal(out, ["one", "four"])

    fn = vectorize(test, "()->()")
    out = fn([["one", "two", "three"], ["four", "five", "six"]])
    assert_array_equal(out, [["o", "t", "t"], ["f", "f", "s"]])


def test_vectorize_coroutine_expand():
    async def test(x):
        return np.array([x for i in range(3)])

    fn = vectorize(test, "()->(s)")
    out = fn(["one", "two"])
    assert_array_equal(out, [["one", "one", "one"], ["two", "two", "two"]])
