import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional

import numpy as np
from numpy.lib.function_base import (
    _calculate_shapes,
    _parse_gufunc_signature,
    _parse_input_dimensions,
    _update_dim_sizes,
)

# Allow nested loops for running in notebook. We don't enable it globally as it
# may interfere with other libraries that use asyncio.
if hasattr(builtins, "__IPYTHON__"):
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        print(
            "Couldn't patch nest_asyncio because it's not installed. Running in the notebook might be have issues"
        )


class vectorize:
    """Returns an object that acts like a function but takes arrays as an input.

    The vectorized function evaluates `func` over successive tuples of the input
    chararrays and returns a single NumPy chararrays or a tuple of NumPy chararrays.

    Its behavior is similar to NumPy's `vectorize` for Python functions: the function
    being vectorized is executed in a `for` loop. Coroutines, however, are executed
    concurrently.

    Part of the code was adapted from `numpy.lib.function_base`.

    """

    def __init__(self, func: Callable, signature: Optional[str] = None):
        self.func = func
        self.signature = signature
        self.is_coroutine_fn = inspect.iscoroutinefunction(func)

        functools.update_wrapper(self, func)

        if signature is not None:
            # Parse the signature string into a Python data structure.
            # For instance "(m),(s)->(s,m)" becomes `([(m,),(s,)],[(s,m)])`.
            self._in_and_out_core_dimensions = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dimensions = None

    def __call__(self, *args, **kwargs):
        """Call the vectorized function."""
        if not args and not kwargs:
            return self.call_thunk()
        elif self.signature is not None:
            return self.call_with_signature(*args, **kwargs)
        else:
            return self.call_no_signature(*args, **kwargs)

    def call_thunk(self):
        """Call a vectorized thunk.

        Thunks have no arguments and can thus be called directly.

        """
        if self.is_coroutine_fn:
            loop = asyncio.new_event_loop()
            try:
                outputs = loop.run_until_complete(self.func())
            finally:
                loop.close()
        else:
            outputs = self.func()

        return outputs

    def call_no_signature(self, *args, **kwargs):
        """Call functions and coroutines when no signature is specified.

        When no signature is specified we assume that all of the function's
        inputs and outputs are scalars (core dimension of zero). We first
        broadcast the input arrays, then iteratively apply the function over the
        elements of the broadcasted arrays and finally reshape the results to
        match the input shape.

        Functions are executed in a for loop, coroutines are executed
        concurrently.

        """
        # Convert args and kwargs to arrays
        args = [np.array(arg) for arg in args]
        kwargs = {key: np.array(value) for key, value in kwargs.items()}

        # Broadcast args and kwargs
        broadcast_shape = np.broadcast(*args, *list(kwargs.values())).shape
        args = [np.broadcast_to(arg, broadcast_shape) for arg in args]
        kwargs = {
            key: np.broadcast_to(value, broadcast_shape)
            for key, value in kwargs.items()
        }

        # Execute functions in a loop, and coroutines concurrently
        if self.is_coroutine_fn:
            outputs = self.vectorize_call_coroutine(broadcast_shape, args, kwargs)
        else:
            outputs = self.vectorize_call(broadcast_shape, args, kwargs)

        # `outputs` is a flat array or a tuple of flat arrays. We reshape the arrays
        # to match the input shape.
        outputs = [
            results if isinstance(results, tuple) else (results,) for results in outputs
        ]
        outputs = tuple(
            [np.asarray(x).reshape(broadcast_shape).squeeze() for x in zip(*outputs)]
        )
        outputs = tuple([x.item() if np.ndim(x) == 0 else x for x in outputs])

        n_results = len(list(outputs))

        return outputs[0] if n_results == 1 else outputs

    def call_with_signature(self, *args, **kwargs):
        """Call functions and coroutines when a signature is specified."""
        input_core_dims, output_core_dims = self._in_and_out_core_dimensions

        # Make sure that the numbers of arguments passed is compatible with
        # the signature.
        num_args = len(args) + len(kwargs)
        if num_args != len(input_core_dims):
            raise TypeError(
                "wrong number of positional arguments: "
                "expected %r, got %r" % (len(input_core_dims), len(args))
            )

        # Convert args and kwargs to arrays
        args = [np.asarray(arg) for arg in args]
        kwargs = {key: np.array(value) for key, value in kwargs.items()}

        # Find the arguments' broadcast shape, and map placeholder
        # variables in the signature to the number of dimensions
        # they correspond to given the arguments.
        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args + list(kwargs.values()), input_core_dims
        )

        # Calculate the shape to which each of the arguments should be broadcasted
        # and reshape them accordingly.
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes, input_core_dims)
        args = [
            np.broadcast_to(arg, shape, subok=True)
            for arg, shape in zip(args, input_shapes)
        ]
        kwargs = {
            key: np.broadcast_to(value, broadcast_shape)
            for key, value in kwargs.items()
        }

        n_out = len(output_core_dims)

        if self.is_coroutine_fn:
            outputs = self.vectorize_call_coroutine(broadcast_shape, args, kwargs)
        else:
            outputs = self.vectorize_call(broadcast_shape, args, kwargs)

        outputs = [
            results if isinstance(results, tuple) else (results,) for results in outputs
        ]

        flat_outputs = list(zip(*outputs))
        n_results = len(flat_outputs)

        if n_out != n_results:
            raise ValueError(
                f"wrong number of outputs from the function, expected {n_out}, got {n_results}"
            )

        # The number of dimensions of the outputs are not necessarily known in
        # advance. The following iterates over the results and updates the
        # number of dimensions of the outputs accordingly.
        for results, core_dims in zip(flat_outputs, output_core_dims):
            for result in results:
                _update_dim_sizes(dim_sizes, result, core_dims)

        # Calculate the shape to which each of the outputs should be broadcasted
        # and reshape them.
        shapes = _calculate_shapes(broadcast_shape, dim_sizes, output_core_dims)
        outputs = tuple(
            [
                np.hstack(results).reshape(shape).squeeze()
                for shape, results in zip(shapes, zip(*outputs))
            ]
        )
        outputs = tuple([x.item() if np.ndim(x) == 0 else x for x in outputs])

        return outputs[0] if n_results == 1 else outputs

    def vectorize_call(self, broadcast_shape, args, kwargs):
        """Run the function in a for loop.

        A possible extension would be to parallelize the calls.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        args
            The function's broadcasted arguments.
        kwargs
            The function's broadcasted keyword arguments.

        """
        outputs = []
        for index in np.ndindex(*broadcast_shape):
            current_args = tuple(arg[index] for arg in args)
            current_kwargs = {key: value[index] for key, value in kwargs.items()}
            outputs.append(self.func(*current_args, **current_kwargs))

        return outputs

    def vectorize_call_coroutine(self, broadcast_shape, args, kwargs):
        """Run coroutines concurrently.

        Creates as many tasks as needed and executes them in a new event
        loop.

        Parameters
        ----------
        broadcast_shape
            The brodcast shape of the input arrays.
        args
            The function's broadcasted arguments.
        kwargs
            The function's broadcasted keyword arguments.

        """

        async def create_and_gather_tasks():
            tasks = []
            for index in np.ndindex(*broadcast_shape):
                current_args = tuple(arg[index] for arg in args)
                current_kwargs = {key: value[index] for key, value in kwargs.items()}
                tasks.append(self.func(*current_args, **current_kwargs))

            outputs = await asyncio.gather(*tasks)

            return outputs

        loop = asyncio.new_event_loop()
        try:
            outputs = loop.run_until_complete(create_and_gather_tasks())
        finally:
            loop.close()

        return outputs


def _update_arrays_type(arrays, results):
    """Update the dtype of arrays.

    String arrays contain strings of fixed length. Here they are initialized with
    the type of the first results, so that if the next results contain longer
    strings they will be truncated when added to the output arrays. Here we
    update the type if the current results contain longer strings than in the
    current output array.

    Parameters
    ----------
    arrays
        Arrays that contain the vectorized function's results.
    results
        The current output of the function being vectorized.

    """

    updated_arrays = []
    for array, result in zip(arrays, results):
        if array.dtype.type == np.str_:
            if array.dtype < np.array(result).dtype:
                array = array.astype(np.array(result).dtype)

        updated_arrays.append(array)

    return tuple(updated_arrays)
